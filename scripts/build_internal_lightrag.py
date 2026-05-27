#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("build_internal_lightrag")

DEFAULT_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw")
DEFAULT_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal_lightrag")
DEFAULT_WORKSPACE = "internal_lightrag"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
DEFAULT_VLLM_API_BASE = "http://localhost:8001/v1"
DEFAULT_EMBEDDING_MODEL = "/data/h50056787/models/bge-m3"
DEFAULT_RERANK_MODEL = "/data/h50056787/models/bge-reranker-v2-m3"
DEFAULT_EXTENSIONS = (
    ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,"
    ".doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md"
)
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".mdx",
    ".html",
    ".htm",
    ".tex",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".csv",
    ".log",
    ".conf",
    ".ini",
    ".properties",
    ".sql",
    ".bat",
    ".sh",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".py",
    ".java",
    ".js",
    ".ts",
    ".swift",
    ".go",
    ".rb",
    ".php",
    ".css",
    ".scss",
    ".less",
}


@dataclass(frozen=True)
class BuildConfig:
    raw_dir: Path
    storage_root: Path
    working_dir: Path
    input_dir: Path
    report_dir: Path
    workspace: str
    max_files: int | None
    max_parallel_insert: int
    recursive: bool
    extensions: tuple[str, ...]
    dry_run: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _default_env_file() -> Path:
    configured = _repo_root() / "configs" / "local-qwen3vl.env"
    if configured.exists():
        return configured
    return _repo_root() / ".env"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _parse_extensions(value: str) -> tuple[str, ...]:
    exts: list[str] = []
    for item in value.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.append(ext)
    return tuple(dict.fromkeys(exts))


def _setup_logging(report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = report_dir / "build_internal_lightrag.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return log_file


def _scan_files(raw_dir: Path, extensions: tuple[str, ...], recursive: bool) -> list[Path]:
    iterator = raw_dir.rglob("*") if recursive else raw_dir.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda path: path.as_posix().lower())


def _file_sha(path: Path, limit: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    digest.update(str(path).encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:10]


def _stage_file(source: Path, input_dir: Path, workspace: str) -> Path:
    target_dir = input_dir / workspace
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem[:120] or "document"
    target = target_dir / f"{stem}__{_file_sha(source)}{source.suffix.lower()}"
    if target.exists():
        return target
    try:
        target.symlink_to(source)
        return target
    except Exception:
        pass
    try:
        os.link(source, target)
        return target
    except Exception:
        pass
    shutil.copy2(source, target)
    return target


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _build_config(args: argparse.Namespace) -> BuildConfig:
    storage_root = Path(args.storage_root).expanduser() if args.storage_root else DEFAULT_STORAGE_ROOT
    workspace = args.workspace or os.getenv("WORKSPACE", DEFAULT_WORKSPACE)
    working_dir = (
        Path(args.working_dir).expanduser()
        if args.working_dir
        else _env_path("WORKING_DIR", storage_root / "rag_workspace")
    )
    input_dir = (
        Path(args.input_dir).expanduser()
        if args.input_dir
        else _env_path("INPUT_DIR", storage_root / "inputs")
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_dir = (
        Path(args.report_dir).expanduser()
        if args.report_dir
        else _env_path("REPORT_DIR", storage_root / "reports") / timestamp
    )
    return BuildConfig(
        raw_dir=Path(args.raw_dir).expanduser(),
        storage_root=storage_root,
        working_dir=working_dir,
        input_dir=input_dir,
        report_dir=report_dir,
        workspace=workspace,
        max_files=args.max_files,
        max_parallel_insert=max(1, int(args.max_concurrent_files)),
        recursive=not args.no_recursive,
        extensions=_parse_extensions(args.extensions),
        dry_run=bool(args.dry_run),
    )


def _apply_runtime_env(config: BuildConfig) -> None:
    defaults = {
        "WORKSPACE": config.workspace,
        "WORKING_DIR": config.working_dir.as_posix(),
        "INPUT_DIR": config.input_dir.as_posix(),
        "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
        "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
        "LIGHTRAG_GRAPH_STORAGE": "Neo4JStorage",
        "LIGHTRAG_VECTOR_STORAGE": "QdrantVectorDBStorage",
        "MAX_PARALLEL_INSERT": str(config.max_parallel_insert),
        "MAX_SOURCE_IDS_PER_ENTITY": "99999",
        "MAX_SOURCE_IDS_PER_RELATION": "99999",
        "SOURCE_IDS_LIMIT_METHOD": "FIFO",
        "VLLM_API_BASE": DEFAULT_VLLM_API_BASE,
        "VLLM_API_KEY": "EMPTY",
        "LLM_MODEL_NAME": DEFAULT_MODEL,
        "RAGANYTHING_EMBEDDING_MODEL_PATH": DEFAULT_EMBEDDING_MODEL,
        "RAGANYTHING_RERANK_MODEL_PATH": DEFAULT_RERANK_MODEL,
        "RAGANYTHING_EMBEDDING_DIM": "1024",
        "RAGANYTHING_EMBEDDING_BATCH_NUM": "32",
        "RAGANYTHING_RERANK_BATCH_SIZE": "8",
        "VLM_PROCESS_ENABLE": "true",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ["WORKSPACE"] = config.workspace
    os.environ["WORKING_DIR"] = config.working_dir.as_posix()
    os.environ["INPUT_DIR"] = config.input_dir.as_posix()
    os.environ["MAX_PARALLEL_INSERT"] = str(config.max_parallel_insert)


def _make_llm_func():
    from lightrag.llm.openai import openai_complete_if_cache

    model = os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL)
    api_base = os.getenv("VLLM_API_BASE", DEFAULT_VLLM_API_BASE)
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    timeout = _env_int("LLM_TIMEOUT", 1800)

    async def local_qwen_complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=api_base,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

    return local_qwen_complete


def _make_embedding_func():
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from lightrag.utils import EmbeddingFunc

    model_path = os.getenv("RAGANYTHING_EMBEDDING_MODEL_PATH", DEFAULT_EMBEDDING_MODEL)
    device = os.getenv("RAGANYTHING_DEVICE", "cuda:0")
    dim = _env_int("RAGANYTHING_EMBEDDING_DIM", 1024)
    batch_num = _env_int("RAGANYTHING_EMBEDDING_BATCH_NUM", _env_int("EMBEDDING_BATCH_NUM", 32))
    model_holder: dict[str, SentenceTransformer] = {}

    def get_model() -> SentenceTransformer:
        model = model_holder.get("model")
        if model is None:
            LOGGER.info("Loading embedding model: %s on %s", model_path, device)
            model = SentenceTransformer(model_path, device=device)
            model_holder["model"] = model
        return model

    async def compute(texts: list[str], **_: Any) -> np.ndarray:
        if not texts:
            return np.empty((0, dim), dtype=np.float32)

        def encode() -> np.ndarray:
            model = get_model()
            result = model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=max(1, min(len(texts), batch_num)),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.asarray(result, dtype=np.float32)

        return await asyncio.to_thread(encode)

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=_env_int("EMBEDDING_TOKEN_LIMIT", 8192),
        func=compute,
        model_name=Path(model_path).name or "local_embedding",
    )


def _make_rerank_func():
    import numpy as np
    from sentence_transformers import CrossEncoder

    model_path = os.getenv("RAGANYTHING_RERANK_MODEL_PATH", DEFAULT_RERANK_MODEL)
    device = os.getenv("RAGANYTHING_DEVICE", "cuda:0")
    batch_size = _env_int("RAGANYTHING_RERANK_BATCH_SIZE", 8)
    model_holder: dict[str, CrossEncoder] = {}

    def get_model() -> CrossEncoder:
        model = model_holder.get("model")
        if model is None:
            LOGGER.info("Loading reranker model: %s on %s", model_path, device)
            model = CrossEncoder(model_path, device=device)
            model_holder["model"] = model
        return model

    async def rerank(
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **_: Any,
    ) -> list[dict[str, float | int]]:
        if not documents:
            return []

        def predict() -> list[dict[str, float | int]]:
            model = get_model()
            pairs = [(query, doc) for doc in documents]
            raw_scores = model.predict(pairs, batch_size=max(1, batch_size))
            scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
            ranked = sorted(
                enumerate(scores.tolist()), key=lambda item: item[1], reverse=True
            )
            if top_n is not None:
                ranked = ranked[: max(0, int(top_n))]
            return [
                {"index": int(index), "relevance_score": float(score)}
                for index, score in ranked
            ]

        return await asyncio.to_thread(predict)

    return rerank


async def _enqueue_file(rag: Any, source: Path, config: BuildConfig, track_id: str) -> dict[str, Any]:
    from lightrag.constants import (
        FULL_DOCS_FORMAT_PENDING_PARSE,
        PARSER_ENGINE_LEGACY,
        PROCESS_OPTION_CHUNK_FIXED,
    )
    from lightrag.parser.routing import resolve_file_parser_directives

    staged = _stage_file(source, config.input_dir, config.workspace)
    engine, process_options = resolve_file_parser_directives(staged)
    process_options = process_options or PROCESS_OPTION_CHUNK_FIXED
    started = time.time()
    if engine == PARSER_ENGINE_LEGACY:
        if staged.suffix.lower() not in TEXT_EXTENSIONS:
            raise RuntimeError(
                f"{source} resolved to legacy parser, but this script only uses "
                "legacy mode for UTF-8 text files. Configure LIGHTRAG_PARSER "
                "to route rich documents to mineru or docling."
            )
        content = staged.read_text(encoding="utf-8")
        await rag.apipeline_enqueue_documents(
            input=content,
            file_paths=staged.name,
            track_id=track_id,
            parse_engine=engine,
            process_options=process_options,
        )
    else:
        await rag.apipeline_enqueue_documents(
            input="",
            file_paths=str(staged),
            track_id=track_id,
            docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
            parse_engine=engine,
            process_options=process_options,
        )
    return {
        "source": source.as_posix(),
        "staged": staged.as_posix(),
        "parser": engine,
        "process_options": process_options,
        "enqueue_seconds": time.time() - started,
    }


async def _collect_doc_status(rag: Any) -> dict[str, Any]:
    doc_status = getattr(rag, "doc_status", None)
    if doc_status is None:
        return {"documents": [], "count": 0}
    getter = getattr(doc_status, "get_docs_paginated", None)
    if not callable(getter):
        return {"documents": [], "count": 0}
    rows, total = await getter(page=1, page_size=1000, sort_field="updated_at", sort_direction="desc")
    documents = []
    for row in rows:
        if isinstance(row, tuple) and len(row) == 2:
            doc_id, payload = row
        else:
            payload = row
            doc_id = getattr(payload, "id", "")
        if hasattr(payload, "__dict__"):
            payload = dict(payload.__dict__)
        documents.append({"doc_id": str(doc_id), "payload": payload})
    return {"count": len(documents), "total": total, "documents": documents}


async def _run_build(config: BuildConfig, files: list[Path]) -> dict[str, Any]:
    from lightrag import LightRAG
    from lightrag.llm_roles import RoleLLMConfig
    from lightrag.utils import generate_track_id

    llm_func = _make_llm_func()
    embedding_func = _make_embedding_func()
    rerank_func = _make_rerank_func()
    track_id = generate_track_id("internal_lightrag")
    config.working_dir.mkdir(parents=True, exist_ok=True)
    config.input_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=config.working_dir.as_posix(),
        workspace=config.workspace,
        llm_model_func=llm_func,
        llm_model_name=os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL),
        llm_model_max_async=_env_int("MAX_ASYNC", 4),
        llm_model_kwargs={},
        default_llm_timeout=_env_int("LLM_TIMEOUT", 1800),
        embedding_func=embedding_func,
        embedding_func_max_async=_env_int("EMBEDDING_FUNC_MAX_ASYNC", 4),
        embedding_batch_num=_env_int("EMBEDDING_BATCH_NUM", 32),
        rerank_model_func=rerank_func,
        rerank_model_max_async=_env_int("MAX_ASYNC_RERANK", 2),
        default_rerank_timeout=_env_int("RERANK_TIMEOUT", 120),
        min_rerank_score=float(os.getenv("MIN_RERANK_SCORE", "0.3")),
        max_parallel_insert=config.max_parallel_insert,
        chunk_token_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap_token_size=_env_int("CHUNK_OVERLAP_SIZE", 100),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "Neo4JStorage"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "QdrantVectorDBStorage"),
        vlm_process_enable=os.getenv("VLM_PROCESS_ENABLE", "true").lower() in {"1", "true", "yes", "on"},
        role_llm_configs={
            "vlm": RoleLLMConfig(
                func=llm_func,
                max_async=_env_int("VLM_MAX_ASYNC_LLM", 2),
                timeout=_env_int("LLM_TIMEOUT", 1800),
                metadata={
                    "binding": "openai",
                    "model": os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL),
                    "host": os.getenv("VLLM_API_BASE", DEFAULT_VLLM_API_BASE),
                },
            )
        },
    )

    enqueued: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.time()
    try:
        await rag.initialize_storages()
        for source in files:
            try:
                record = await _enqueue_file(rag, source, config, track_id)
                enqueued.append(record)
                LOGGER.info(
                    "Enqueued %s parser=%s options=%s",
                    source.name,
                    record["parser"],
                    record["process_options"],
                )
            except Exception as exc:
                LOGGER.exception("Failed to enqueue %s", source)
                failures.append({"source": source.as_posix(), "error": str(exc)})
        if enqueued:
            await rag.apipeline_process_enqueue_documents()
        documents = await _collect_doc_status(rag)
    finally:
        await rag.finalize_storages()

    return {
        "track_id": track_id,
        "elapsed_seconds": time.time() - start,
        "enqueued_count": len(enqueued),
        "failed_enqueue_count": len(failures),
        "enqueued": enqueued,
        "failures": failures,
        "documents": documents,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an internal workspace with official LightRAG and local RAG_LUND models."
    )
    parser.add_argument("--env-file", default=str(_default_env_file()))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--storage-root", default=str(DEFAULT_STORAGE_ROOT))
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-concurrent-files", type=int, default=2)
    parser.add_argument("--extensions", default=os.getenv("LIGHTRAG_BUILD_EXTENSIONS", DEFAULT_EXTENSIONS))
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_env_file(Path(args.env_file).expanduser())
    config = _build_config(args)
    _apply_runtime_env(config)
    log_file = _setup_logging(config.report_dir)
    LOGGER.info("Log file: %s", log_file)

    if not config.raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {config.raw_dir}")
    files = _scan_files(config.raw_dir, config.extensions, config.recursive)
    if config.max_files is not None:
        files = files[: max(0, int(config.max_files))]

    summary_base = {
        "workspace": config.workspace,
        "raw_dir": config.raw_dir.as_posix(),
        "working_dir": config.working_dir.as_posix(),
        "input_dir": config.input_dir.as_posix(),
        "report_dir": config.report_dir.as_posix(),
        "file_count": len(files),
        "files": [path.as_posix() for path in files],
        "max_parallel_insert": config.max_parallel_insert,
        "dry_run": config.dry_run,
    }

    if config.dry_run:
        _write_json(config.report_dir / "build_summary.json", summary_base)
        LOGGER.info("Dry run complete: %d files", len(files))
        return 0

    result = asyncio.run(_run_build(config, files))
    summary = {**summary_base, **result}
    _write_json(config.report_dir / "build_summary.json", summary)
    _write_json(config.report_dir / "failed_files.json", result.get("failures", []))
    _write_json(config.report_dir / "documents_status.json", result.get("documents", {}))
    LOGGER.info("Build finished. Summary: %s", config.report_dir / "build_summary.json")
    return 1 if result.get("failed_enqueue_count") else 0


if __name__ == "__main__":
    raise SystemExit(main())
