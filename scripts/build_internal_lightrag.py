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
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RAW_DIR = Path("/data/y50056788/Yaliang/datasets_raw")
DEFAULT_STORAGE_ROOT = Path("/data/y50056788/Yaliang/internal_lightrag")
DEFAULT_WORKSPACE = "internal_lightrag"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
DEFAULT_VLLM_API_BASE = "http://localhost:8001/v1"
DEFAULT_TOKENIZER_MODEL = "/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8"
DEFAULT_EMBEDDING_MODEL = "/data/h50056787/models/bge-m3"
DEFAULT_RERANK_MODEL = "/data/h50056787/models/bge-reranker-v2-m3"
DEFAULT_DEV_LIBREOFFICE_PDF_ROOT = Path(
    "/data/y50056788/Yaliang/internal/output/internal"
)
DEFAULT_MAX_ASYNC = 16
DEFAULT_EMBEDDING_BATCH_NUM = 4
LOCAL_HYBRID_VECTOR_STORAGE = "QdrantHybridBM25VectorDBStorage"
DEFAULT_EXTENSIONS = (
    ".pdf,.jpg,.jpeg,.png,.bmp,.gif,.webp,"
    ".doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md"
)
DEFAULT_LIGHTRAG_PARSER = (
    "pdf:mineru-iteP,doc:mineru-iteP,docx:mineru-iteP,"
    "ppt:mineru-iteP,pptx:mineru-iteP,xls:mineru-iteP,"
    "xlsx:mineru-iteP,png:mineru-iteP,jpg:mineru-iteP,"
    "jpeg:mineru-iteP,bmp:mineru-iteP,gif:mineru-iteP,webp:mineru-iteP,"
    "txt:legacy-F,md:legacy-F"
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
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
UNSUPPORTED_MINERU_SUFFIXES = {"tif", "tiff"}


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
    enable_build_rerank: bool
    enable_query_rerank: bool
    query: str | None
    query_file: Path | None
    query_only: bool
    query_mode: str
    top_k: int | None
    chunk_top_k: int | None
    reuse_dev_libreoffice_pdfs: bool = True
    dev_libreoffice_pdf_root: Path = DEFAULT_DEV_LIBREOFFICE_PDF_ROOT


def _repo_root() -> Path:
    return REPO_ROOT


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _normalize_lightrag_parser_rules(value: str) -> str:
    rules: list[str] = []
    for raw_item in value.replace(";", ",").split(","):
        item = raw_item.strip()
        if not item or ":" not in item:
            continue
        pattern, engine = item.split(":", 1)
        pattern = pattern.strip().lower()
        if pattern.startswith("*."):
            pattern = pattern[2:]
        elif pattern.startswith("."):
            pattern = pattern[1:]
        engine_name = engine.strip().split("-", 1)[0].lower()
        if engine_name == "mineru" and pattern in UNSUPPORTED_MINERU_SUFFIXES:
            continue
        rules.append(f"{pattern}:{engine.strip()}")
    return ",".join(rules) if rules else value


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


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem) or "file"


def _converted_pdf_paths(source: Path, output_root: Path) -> tuple[Path, ...]:
    preferred_pdf = output_root / _safe_stem(source) / f"{source.stem}.pdf"
    legacy_root_pdf = output_root / f"{source.stem}.pdf"
    if preferred_pdf == legacy_root_pdf:
        return (preferred_pdf,)
    return (preferred_pdf, legacy_root_pdf)


def _valid_converted_pdf(pdf_path: Path, source_path: Path) -> bool:
    try:
        if not pdf_path.exists() or not pdf_path.is_file() or pdf_path.stat().st_size <= 0:
            return False
        if pdf_path.stat().st_mtime < source_path.stat().st_mtime:
            LOGGER.warning(
                "Reusing converted PDF older than source file=%s pdf=%s. "
                "Delete the PDF to force reconversion after source content changes.",
                source_path.name,
                pdf_path,
            )
        return True
    except OSError:
        return False


def _find_valid_converted_pdf(source: Path, output_root: Path) -> Path | None:
    for pdf_path in _converted_pdf_paths(source, output_root):
        if _valid_converted_pdf(pdf_path, source):
            return pdf_path
    return None


def _resolve_effective_source(source: Path, config: BuildConfig) -> dict[str, Any]:
    candidates: tuple[Path, ...] = ()
    converted_pdf: Path | None = None
    if config.reuse_dev_libreoffice_pdfs and source.suffix.lower() in OFFICE_EXTENSIONS:
        candidates = _converted_pdf_paths(source, config.dev_libreoffice_pdf_root)
        converted_pdf = _find_valid_converted_pdf(source, config.dev_libreoffice_pdf_root)
    if converted_pdf is None:
        return {
            "source": source,
            "effective_source": source,
            "reused_converted_pdf": False,
            "converted_pdf": None,
            "converted_pdf_candidates": [path.as_posix() for path in candidates],
        }
    return {
        "source": source,
        "effective_source": converted_pdf,
        "reused_converted_pdf": True,
        "converted_pdf": converted_pdf.as_posix(),
        "converted_pdf_candidates": [path.as_posix() for path in candidates],
    }


def _stage_file(
    source: Path,
    input_dir: Path,
    workspace: str,
    *,
    identity_source: Path | None = None,
    suffix: str | None = None,
) -> Path:
    target_dir = input_dir / workspace
    target_dir.mkdir(parents=True, exist_ok=True)
    identity = identity_source or source
    target_suffix = suffix if suffix is not None else source.suffix.lower()
    stem = identity.stem[:120] or "document"
    target = target_dir / f"{stem}__{_file_sha(identity)}{target_suffix.lower()}"
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


def _prepare_staged_input(source: Path, config: BuildConfig) -> dict[str, Any]:
    resolved = _resolve_effective_source(source, config)
    effective_source = Path(resolved["effective_source"])
    staged = _stage_file(
        effective_source,
        config.input_dir,
        config.workspace,
        identity_source=source,
        suffix=effective_source.suffix.lower(),
    )
    return {
        **resolved,
        "effective_source": effective_source.as_posix(),
        "staged": staged.as_posix(),
        "file_path": staged.name,
    }


def _build_file_reuse_preview(config: BuildConfig, files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in files:
        resolved = _resolve_effective_source(source, config)
        effective_source = Path(resolved["effective_source"])
        records.append(
            {
                "source": source.as_posix(),
                "effective_source": effective_source.as_posix(),
                "would_reuse_converted_pdf": bool(resolved["reused_converted_pdf"]),
                "reused_converted_pdf": bool(resolved["reused_converted_pdf"]),
                "converted_pdf": resolved["converted_pdf"],
                "converted_pdf_candidates": resolved["converted_pdf_candidates"],
                "staged_basename": (
                    f"{(source.stem[:120] or 'document')}__"
                    f"{_file_sha(source)}{effective_source.suffix.lower()}"
                ),
            }
        )
    return records


def _reuse_summary(
    records: list[dict[str, Any]], config: BuildConfig | None = None
) -> dict[str, Any]:
    reused = [
        record
        for record in records
        if record.get("reused_converted_pdf") or record.get("would_reuse_converted_pdf")
    ]
    enabled = (
        config.reuse_dev_libreoffice_pdfs
        if config is not None
        else _env_bool("REUSE_DEV_LIBREOFFICE_PDFS", True)
    )
    pdf_root = (
        config.dev_libreoffice_pdf_root.as_posix()
        if config is not None
        else os.getenv(
            "DEV_LIBREOFFICE_PDF_ROOT", DEFAULT_DEV_LIBREOFFICE_PDF_ROOT.as_posix()
        )
    )
    return {
        "enabled": enabled,
        "dev_libreoffice_pdf_root": pdf_root,
        "file_count": len(records),
        "reused_converted_pdf_count": len(reused),
        "fallback_to_original_count": len(records) - len(reused),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _doc_payload_dict(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        data = dict(payload)
    elif hasattr(payload, "__dict__"):
        data = dict(payload.__dict__)
    else:
        data = {}
    status = data.get("status")
    if hasattr(status, "value"):
        data["status"] = status.value
    elif status is not None:
        data["status"] = str(status)
    return data


def _doc_status_name(payload: dict[str, Any]) -> str:
    status = payload.get("status", "")
    if hasattr(status, "value"):
        return str(status.value)
    return str(status or "").lower()


def _existing_doc_record_for_source(
    source: Path, existing_by_file: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    return existing_by_file.get(source.name)


async def _collect_existing_doc_index(rag: Any) -> dict[str, dict[str, Any]]:
    doc_status = getattr(rag, "doc_status", None)
    if doc_status is None:
        return {}
    rows: dict[str, Any] = {}
    getter = getattr(doc_status, "get_docs_by_statuses", None)
    if callable(getter):
        from lightrag.base import DocStatus

        rows = await getter(list(DocStatus))
    else:
        paginated = getattr(doc_status, "get_docs_paginated", None)
        if callable(paginated):
            page_rows, _ = await paginated(
                page=1,
                page_size=_env_int("DOC_STATUS_SCAN_PAGE_SIZE", 10000),
                sort_field="updated_at",
                sort_direction="desc",
            )
            for row in page_rows:
                if isinstance(row, tuple) and len(row) == 2:
                    rows[str(row[0])] = row[1]
                else:
                    rows[str(getattr(row, "id", ""))] = row

    existing: dict[str, dict[str, Any]] = {}
    for doc_id, payload in rows.items():
        data = _doc_payload_dict(payload)
        file_path = str(data.get("file_path") or "").strip()
        if not file_path:
            continue
        data["doc_id"] = str(doc_id)
        existing[Path(file_path).name] = data
    return existing


def _build_compact_summary(
    summary_base: dict[str, Any], result: dict[str, Any] | None = None
) -> dict[str, Any]:
    result = result or {}
    documents = list(result.get("documents", {}).get("documents", []))
    file_records: list[dict[str, Any]] = []

    relevant_files: set[str] = set()
    for record in result.get("enqueued", []):
        staged = record.get("staged")
        if staged:
            relevant_files.add(Path(str(staged)).name)
        file_records.append(
            {
                "source": record.get("source"),
                "effective_source": record.get("effective_source"),
                "staged": staged,
                "file_path": Path(str(staged)).name if staged else record.get("file_path"),
                "status": "enqueued",
                "reused_converted_pdf": bool(record.get("reused_converted_pdf")),
                "converted_pdf": record.get("converted_pdf"),
            }
        )
    for record in result.get("skipped_existing", []):
        file_path = record.get("file_path") or record.get("staged")
        if file_path:
            relevant_files.add(Path(str(file_path)).name)
        file_records.append(
            {
                "source": record.get("source"),
                "effective_source": record.get("effective_source"),
                "staged": record.get("staged"),
                "file_path": Path(str(file_path)).name if file_path else None,
                "status": "skipped_existing",
                "doc_id": record.get("doc_id"),
                "doc_status": record.get("status"),
                "reused_converted_pdf": bool(record.get("reused_converted_pdf")),
                "converted_pdf": record.get("converted_pdf"),
            }
        )

    selected_docs: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    failed_files: list[dict[str, Any]] = []
    for failure in result.get("failures", []):
        failed_files.append(
            {
                "file": failure.get("source", "unknown"),
                "stage": "enqueue",
                "error": failure.get("error", ""),
            }
        )
        file_records.append(
            {
                "source": failure.get("source"),
                "effective_source": failure.get("effective_source"),
                "staged": failure.get("staged"),
                "file_path": failure.get("file_path"),
                "status": "failed_enqueue",
                "reused_converted_pdf": bool(failure.get("reused_converted_pdf")),
                "converted_pdf": failure.get("converted_pdf"),
                "error": failure.get("error", ""),
            }
        )

    for doc in documents:
        payload = _doc_payload_dict(doc.get("payload", {}))
        file_path = str(payload.get("file_path") or "")
        if relevant_files and Path(file_path).name not in relevant_files:
            continue
        status = _doc_status_name(payload)
        status_counts[status] = status_counts.get(status, 0) + 1
        selected_docs.append(
            {
                "doc_id": str(doc.get("doc_id", "")),
                "file_path": file_path,
                "status": status,
                "track_id": payload.get("track_id"),
                "chunks_count": payload.get("chunks_count"),
                "error_msg": payload.get("error_msg"),
                "metadata": payload.get("metadata"),
            }
        )
        if status == "failed":
            failed_files.append(
                {
                    "file": file_path or str(doc.get("doc_id", "")),
                    "doc_id": str(doc.get("doc_id", "")),
                    "stage": "pipeline",
                    "error": payload.get("error_msg") or "",
                }
            )

    succeeded_count = status_counts.get("processed", 0)
    failed_count = len(failed_files)
    in_progress_count = sum(
        status_counts.get(status, 0)
        for status in ("pending", "parsing", "analyzing", "processing", "preprocessed")
    )
    if not file_records:
        file_records = list(summary_base.get("file_records", []))
    reuse = result.get("reuse") or summary_base.get("reuse", {})
    return {
        "workspace": summary_base.get("workspace"),
        "raw_dir": summary_base.get("raw_dir"),
        "working_dir": summary_base.get("working_dir"),
        "input_dir": summary_base.get("input_dir"),
        "report_dir": summary_base.get("report_dir"),
        "file_count": summary_base.get("file_count", 0),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "in_progress_count": in_progress_count,
        "enqueued_count": result.get("enqueued_count", 0),
        "skipped_existing_count": result.get("skipped_existing_count", 0),
        "failed_enqueue_count": result.get("failed_enqueue_count", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "failed_files": failed_files,
        "reuse": reuse,
        "file_records": file_records,
        "documents": selected_docs,
        "settings": summary_base.get("settings", {}),
    }


def _resolve_query_text(args: argparse.Namespace) -> str | None:
    if args.query and args.query_file:
        raise ValueError("Use only one of --query or --query-file.")
    if args.query:
        return str(args.query).strip()
    if args.query_file:
        query_path = Path(args.query_file).expanduser()
        return query_path.read_text(encoding="utf-8").strip()
    return None


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
        enable_build_rerank=bool(args.enable_build_rerank),
        enable_query_rerank=bool(args.enable_query_rerank),
        query=_resolve_query_text(args),
        query_file=Path(args.query_file).expanduser() if args.query_file else None,
        query_only=bool(args.query_only),
        query_mode=str(args.query_mode),
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        reuse_dev_libreoffice_pdfs=_env_bool("REUSE_DEV_LIBREOFFICE_PDFS", True),
        dev_libreoffice_pdf_root=_env_path(
            "DEV_LIBREOFFICE_PDF_ROOT", DEFAULT_DEV_LIBREOFFICE_PDF_ROOT
        ),
    )


def _apply_runtime_env(config: BuildConfig) -> None:
    defaults = {
        "WORKSPACE": config.workspace,
        "WORKING_DIR": config.working_dir.as_posix(),
        "INPUT_DIR": config.input_dir.as_posix(),
        "LIGHTRAG_KV_STORAGE": "JsonKVStorage",
        "LIGHTRAG_DOC_STATUS_STORAGE": "JsonDocStatusStorage",
        "LIGHTRAG_GRAPH_STORAGE": "Neo4JStorage",
        "LIGHTRAG_VECTOR_STORAGE": LOCAL_HYBRID_VECTOR_STORAGE,
        "LIGHTRAG_PARSER": DEFAULT_LIGHTRAG_PARSER,
        "MINERU_API_MODE": "local",
        "MINERU_LOCAL_ENDPOINT": "http://127.0.0.1:8000",
        "MINERU_LOCAL_BACKEND": "hybrid-auto-engine",
        "MINERU_LOCAL_PARSE_METHOD": "auto",
        "MINERU_LOCAL_IMAGE_ANALYSIS": "true",
        "QDRANT_ENABLE_SPARSE_BM25": "true",
        "QDRANT_SPARSE_BM25_MODEL": "Qdrant/bm25",
        "QDRANT_RETRIEVAL_MODE": "hybrid",
        "QDRANT_COLLECTION_PREFIX": "local_lightrag_bm25",
        "MAX_PARALLEL_INSERT": str(config.max_parallel_insert),
        "MAX_SOURCE_IDS_PER_ENTITY": "999999",
        "MAX_SOURCE_IDS_PER_RELATION": "999999",
        "SOURCE_IDS_LIMIT_METHOD": "FIFO",
        "REUSE_DEV_LIBREOFFICE_PDFS": "true",
        "DEV_LIBREOFFICE_PDF_ROOT": DEFAULT_DEV_LIBREOFFICE_PDF_ROOT.as_posix(),
        "VLLM_API_BASE": DEFAULT_VLLM_API_BASE,
        "VLLM_API_KEY": "EMPTY",
        "LLM_MODEL_NAME": DEFAULT_MODEL,
        "LLM_TOKENIZER_MODEL_PATH": DEFAULT_TOKENIZER_MODEL,
        "RAGANYTHING_EMBEDDING_MODEL_PATH": DEFAULT_EMBEDDING_MODEL,
        "RAGANYTHING_RERANK_MODEL_PATH": DEFAULT_RERANK_MODEL,
        "RAGANYTHING_EMBEDDING_DIM": "1024",
        "RAGANYTHING_EMBEDDING_BATCH_NUM": str(DEFAULT_EMBEDDING_BATCH_NUM),
        "RAGANYTHING_RERANK_BATCH_SIZE": "8",
        "MAX_ASYNC": str(DEFAULT_MAX_ASYNC),
        "MAX_ASYNC_RERANK": "1",
        "EMBEDDING_FUNC_MAX_ASYNC": "4",
        "EMBEDDING_BATCH_NUM": str(DEFAULT_EMBEDDING_BATCH_NUM),
        "MAX_PARALLEL_PARSE_NATIVE": "2",
        "MAX_PARALLEL_PARSE_MINERU": "1",
        "MAX_PARALLEL_PARSE_DOCLING": "1",
        "MAX_PARALLEL_ANALYZE": "2",
        "MAX_EXTRACT_INPUT_TOKENS": "20480",
        "EMBEDDING_TOKEN_LIMIT": "8192",
        "RERANK_BY_DEFAULT": "true",
        "VLM_PROCESS_ENABLE": "true",
        "VLM_MAX_ASYNC_LLM": "2",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    os.environ["WORKSPACE"] = config.workspace
    os.environ["WORKING_DIR"] = config.working_dir.as_posix()
    os.environ["INPUT_DIR"] = config.input_dir.as_posix()
    os.environ["MAX_PARALLEL_INSERT"] = str(config.max_parallel_insert)
    os.environ["LIGHTRAG_PARSER"] = _normalize_lightrag_parser_rules(
        os.environ.get("LIGHTRAG_PARSER", DEFAULT_LIGHTRAG_PARSER)
    )


def _settings_summary(config: BuildConfig, files: list[Path] | None = None) -> dict[str, Any]:
    keys = (
        "LLM_MODEL_NAME",
        "VLLM_API_BASE",
        "LLM_TOKENIZER_MODEL_PATH",
        "RAGANYTHING_EMBEDDING_MODEL_PATH",
        "RAGANYTHING_RERANK_MODEL_PATH",
        "RAGANYTHING_EMBEDDING_DIM",
        "RAGANYTHING_EMBEDDING_BATCH_NUM",
        "MAX_ASYNC",
        "MAX_ASYNC_RERANK",
        "EMBEDDING_FUNC_MAX_ASYNC",
        "EMBEDDING_BATCH_NUM",
        "MAX_PARALLEL_INSERT",
        "MAX_PARALLEL_PARSE_NATIVE",
        "MAX_PARALLEL_PARSE_MINERU",
        "MAX_PARALLEL_PARSE_DOCLING",
        "MAX_PARALLEL_ANALYZE",
        "VLM_PROCESS_ENABLE",
        "VLM_MAX_ASYNC_LLM",
        "MAX_EXTRACT_INPUT_TOKENS",
        "EMBEDDING_TOKEN_LIMIT",
        "MAX_SOURCE_IDS_PER_ENTITY",
        "MAX_SOURCE_IDS_PER_RELATION",
        "SOURCE_IDS_LIMIT_METHOD",
        "REUSE_DEV_LIBREOFFICE_PDFS",
        "DEV_LIBREOFFICE_PDF_ROOT",
        "RERANK_BY_DEFAULT",
        "LIGHTRAG_PARSER",
        "MINERU_API_MODE",
        "MINERU_LOCAL_ENDPOINT",
        "MINERU_LOCAL_BACKEND",
        "MINERU_LOCAL_PARSE_METHOD",
        "MINERU_LOCAL_IMAGE_ANALYSIS",
        "LIGHTRAG_VECTOR_STORAGE",
        "NEO4J_URI",
        "NEO4J_DATABASE",
        "QDRANT_URL",
        "QDRANT_ENABLE_SPARSE_BM25",
        "QDRANT_SPARSE_BM25_MODEL",
        "QDRANT_RETRIEVAL_MODE",
        "QDRANT_COLLECTION_PREFIX",
        "QDRANT_FASTEMBED_CACHE_DIR",
    )
    return {
        "workspace": config.workspace,
        "raw_dir": config.raw_dir.as_posix(),
        "working_dir": config.working_dir.as_posix(),
        "input_dir": config.input_dir.as_posix(),
        "report_dir": config.report_dir.as_posix(),
        "extensions": config.extensions,
        "recursive": config.recursive,
        "dry_run": config.dry_run,
        "query_only": config.query_only,
        "enable_build_rerank": config.enable_build_rerank,
        "enable_query_rerank": config.enable_query_rerank,
        "query_mode": config.query_mode,
        "top_k": config.top_k,
        "chunk_top_k": config.chunk_top_k,
        "reuse_dev_libreoffice_pdfs": config.reuse_dev_libreoffice_pdfs,
        "dev_libreoffice_pdf_root": config.dev_libreoffice_pdf_root.as_posix(),
        "file_count": len(files) if files is not None else None,
        "env": {key: os.getenv(key) for key in keys if os.getenv(key) is not None},
    }


def _log_settings(config: BuildConfig, files: list[Path] | None = None) -> None:
    summary = _settings_summary(config, files)
    LOGGER.info("Workspace: %s", summary["workspace"])
    LOGGER.info("Raw dir: %s", summary["raw_dir"])
    LOGGER.info("Working dir: %s", summary["working_dir"])
    LOGGER.info("Input dir: %s", summary["input_dir"])
    LOGGER.info("Report dir: %s", summary["report_dir"])
    LOGGER.info(
        "Model: %s via %s",
        summary["env"].get("LLM_MODEL_NAME"),
        summary["env"].get("VLLM_API_BASE"),
    )
    LOGGER.info("Tokenizer: %s", summary["env"].get("LLM_TOKENIZER_MODEL_PATH"))
    LOGGER.info(
        "Embedding: %s dim=%s batch=%s max_async=%s token_limit=%s",
        summary["env"].get("RAGANYTHING_EMBEDDING_MODEL_PATH"),
        summary["env"].get("RAGANYTHING_EMBEDDING_DIM"),
        summary["env"].get("EMBEDDING_BATCH_NUM"),
        summary["env"].get("EMBEDDING_FUNC_MAX_ASYNC"),
        summary["env"].get("EMBEDDING_TOKEN_LIMIT"),
    )
    LOGGER.info(
        "Concurrency: insert=%s llm=%s vlm=%s parse_native=%s parse_mineru=%s parse_docling=%s analyze=%s",
        summary["env"].get("MAX_PARALLEL_INSERT"),
        summary["env"].get("MAX_ASYNC"),
        summary["env"].get("VLM_MAX_ASYNC_LLM"),
        summary["env"].get("MAX_PARALLEL_PARSE_NATIVE"),
        summary["env"].get("MAX_PARALLEL_PARSE_MINERU"),
        summary["env"].get("MAX_PARALLEL_PARSE_DOCLING"),
        summary["env"].get("MAX_PARALLEL_ANALYZE"),
    )
    LOGGER.info(
        "Rerank: build=%s query=%s default_env=%s max_async=%s",
        summary["enable_build_rerank"],
        summary["enable_query_rerank"],
        summary["env"].get("RERANK_BY_DEFAULT"),
        summary["env"].get("MAX_ASYNC_RERANK"),
    )
    LOGGER.info(
        "Qdrant vector storage: %s retrieval_mode=%s sparse_bm25=%s collection_prefix=%s",
        summary["env"].get("LIGHTRAG_VECTOR_STORAGE"),
        summary["env"].get("QDRANT_RETRIEVAL_MODE"),
        summary["env"].get("QDRANT_ENABLE_SPARSE_BM25"),
        summary["env"].get("QDRANT_COLLECTION_PREFIX"),
    )
    LOGGER.info(
        "Token/source caps: max_extract=%s max_source_entity=%s max_source_relation=%s",
        summary["env"].get("MAX_EXTRACT_INPUT_TOKENS"),
        summary["env"].get("MAX_SOURCE_IDS_PER_ENTITY"),
        summary["env"].get("MAX_SOURCE_IDS_PER_RELATION"),
    )
    LOGGER.info(
        "LibreOffice PDF reuse: enabled=%s root=%s",
        summary["reuse_dev_libreoffice_pdfs"],
        summary["dev_libreoffice_pdf_root"],
    )
    LOGGER.info("Parser routing: %s", summary["env"].get("LIGHTRAG_PARSER", "<official default>"))
    if files is not None:
        LOGGER.info("Selected files: %d", len(files))


def _validate_runtime_config() -> None:
    from lightrag.parser.routing import validate_parser_routing_config

    validate_parser_routing_config(os.getenv("LIGHTRAG_PARSER"))


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
    batch_num = _env_int(
        "RAGANYTHING_EMBEDDING_BATCH_NUM",
        _env_int("EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM),
    )
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


def _make_tokenizer():
    from transformers import AutoTokenizer

    from lightrag.utils import Tokenizer

    model_path = os.getenv("LLM_TOKENIZER_MODEL_PATH", DEFAULT_TOKENIZER_MODEL)
    LOGGER.info("Loading tokenizer locally: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    return Tokenizer(Path(model_path).name or os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL), tokenizer)


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


async def _enqueue_staged_file(
    rag: Any,
    source: Path,
    staged: Path,
    config: BuildConfig,
    track_id: str,
    *,
    effective_source: Path | None = None,
    reused_converted_pdf: bool = False,
    converted_pdf: str | None = None,
    converted_pdf_candidates: list[str] | None = None,
) -> dict[str, Any]:
    from lightrag.constants import (
        FULL_DOCS_FORMAT_PENDING_PARSE,
        PARSER_ENGINE_LEGACY,
        PROCESS_OPTION_CHUNK_FIXED,
    )
    from lightrag.parser.routing import resolve_file_parser_directives

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
        "effective_source": (effective_source or staged).as_posix(),
        "staged": staged.as_posix(),
        "file_path": staged.name,
        "reused_converted_pdf": bool(reused_converted_pdf),
        "converted_pdf": converted_pdf,
        "converted_pdf_candidates": converted_pdf_candidates or [],
        "parser": engine,
        "process_options": process_options,
        "enqueue_seconds": time.time() - started,
    }


async def _enqueue_file(rag: Any, source: Path, config: BuildConfig, track_id: str) -> dict[str, Any]:
    stage_info = _prepare_staged_input(source, config)
    return await _enqueue_staged_file(
        rag,
        source,
        Path(stage_info["staged"]),
        config,
        track_id,
        effective_source=Path(stage_info["effective_source"]),
        reused_converted_pdf=bool(stage_info["reused_converted_pdf"]),
        converted_pdf=stage_info["converted_pdf"],
        converted_pdf_candidates=stage_info["converted_pdf_candidates"],
    )


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
        payload = _doc_payload_dict(payload)
        documents.append({"doc_id": str(doc_id), "payload": payload})
    return {"count": len(documents), "total": total, "documents": documents}


def _make_rag(config: BuildConfig, include_reranker: bool) -> Any:
    _register_local_hybrid_bm25_storage()
    from lightrag import LightRAG
    from lightrag.llm_roles import RoleLLMConfig

    llm_func = _make_llm_func()
    embedding_func = _make_embedding_func()
    tokenizer = _make_tokenizer()
    rerank_func = _make_rerank_func() if include_reranker else None
    if include_reranker:
        LOGGER.info("Reranker is enabled for this run.")
    else:
        LOGGER.info("Reranker is disabled for this run; no CrossEncoder will be attached.")

    config.working_dir.mkdir(parents=True, exist_ok=True)
    config.input_dir.mkdir(parents=True, exist_ok=True)

    return LightRAG(
        working_dir=config.working_dir.as_posix(),
        workspace=config.workspace,
        tokenizer=tokenizer,
        llm_model_func=llm_func,
        llm_model_name=os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL),
        llm_model_max_async=_env_int("MAX_ASYNC", DEFAULT_MAX_ASYNC),
        llm_model_kwargs={},
        default_llm_timeout=_env_int("LLM_TIMEOUT", 1800),
        embedding_func=embedding_func,
        embedding_func_max_async=_env_int("EMBEDDING_FUNC_MAX_ASYNC", 4),
        embedding_batch_num=_env_int("EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM),
        rerank_model_func=rerank_func,
        rerank_model_max_async=_env_int("MAX_ASYNC_RERANK", 1),
        default_rerank_timeout=_env_int("RERANK_TIMEOUT", 120),
        min_rerank_score=float(os.getenv("MIN_RERANK_SCORE", "0.3")),
        max_parallel_insert=config.max_parallel_insert,
        chunk_token_size=_env_int("CHUNK_SIZE", 1200),
        chunk_overlap_token_size=_env_int("CHUNK_OVERLAP_SIZE", 100),
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "Neo4JStorage"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", LOCAL_HYBRID_VECTOR_STORAGE),
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


def _register_local_hybrid_bm25_storage() -> None:
    from lightrag import kg

    implementations = kg.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
    if LOCAL_HYBRID_VECTOR_STORAGE not in implementations:
        implementations.append(LOCAL_HYBRID_VECTOR_STORAGE)
    kg.STORAGE_ENV_REQUIREMENTS[LOCAL_HYBRID_VECTOR_STORAGE] = ["QDRANT_URL"]
    kg.STORAGES[LOCAL_HYBRID_VECTOR_STORAGE] = "local_lightrag.qdrant_hybrid_bm25"


async def _run_build(config: BuildConfig, files: list[Path]) -> dict[str, Any]:
    from lightrag.utils import generate_track_id

    track_id = generate_track_id("internal_lightrag")
    rag = _make_rag(config, include_reranker=config.enable_build_rerank)

    enqueued: list[dict[str, Any]] = []
    skipped_existing: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.time()
    try:
        LOGGER.info("Initializing LightRAG storages.")
        await rag.initialize_storages()
        existing_by_file = await _collect_existing_doc_index(rag)
        LOGGER.info(
            "Existing doc_status records indexed by file_path: %d",
            len(existing_by_file),
        )
        for index, source in enumerate(files, start=1):
            stage_info: dict[str, Any] = {}
            try:
                stage_info = _prepare_staged_input(source, config)
                staged = Path(stage_info["staged"])
                if stage_info["reused_converted_pdf"]:
                    LOGGER.info(
                        "Reusing LibreOffice PDF [%d/%d]: source=%s pdf=%s staged=%s",
                        index,
                        len(files),
                        source,
                        stage_info["converted_pdf"],
                        staged,
                    )
                elif source.suffix.lower() in OFFICE_EXTENSIONS and config.reuse_dev_libreoffice_pdfs:
                    LOGGER.info(
                        "No reusable LibreOffice PDF found [%d/%d]: source=%s; falling back to original file.",
                        index,
                        len(files),
                        source,
                    )
                existing = _existing_doc_record_for_source(staged, existing_by_file)
                if existing is not None:
                    record = {
                        "source": source.as_posix(),
                        "effective_source": stage_info["effective_source"],
                        "staged": staged.as_posix(),
                        "file_path": staged.name,
                        "doc_id": existing.get("doc_id", ""),
                        "status": _doc_status_name(existing),
                        "error_msg": existing.get("error_msg"),
                        "reused_converted_pdf": bool(stage_info["reused_converted_pdf"]),
                        "converted_pdf": stage_info["converted_pdf"],
                        "converted_pdf_candidates": stage_info["converted_pdf_candidates"],
                    }
                    skipped_existing.append(record)
                    LOGGER.info(
                        "Skip enqueue [%d/%d]: %s already exists as %s status=%s; "
                        "pipeline resume will handle failed/pending states.",
                        index,
                        len(files),
                        source.name,
                        record["doc_id"],
                        record["status"],
                    )
                    continue
                LOGGER.info(
                    "Enqueue start [%d/%d]: %s size=%d bytes",
                    index,
                    len(files),
                    source,
                    source.stat().st_size,
                )
                record = await _enqueue_staged_file(
                    rag,
                    source,
                    staged,
                    config,
                    track_id,
                    effective_source=Path(stage_info["effective_source"]),
                    reused_converted_pdf=bool(stage_info["reused_converted_pdf"]),
                    converted_pdf=stage_info["converted_pdf"],
                    converted_pdf_candidates=stage_info["converted_pdf_candidates"],
                )
                enqueued.append(record)
                LOGGER.info(
                    "Enqueue done [%d/%d]: %s parser=%s options=%s staged=%s reused_pdf=%s seconds=%.2f",
                    index,
                    len(files),
                    source.name,
                    record["parser"],
                    record["process_options"],
                    record["staged"],
                    record["reused_converted_pdf"],
                    record["enqueue_seconds"],
                )
            except Exception as exc:
                LOGGER.exception("Failed to enqueue %s", source)
                failures.append(
                    {
                        "source": source.as_posix(),
                        "effective_source": stage_info.get("effective_source"),
                        "staged": stage_info.get("staged"),
                        "file_path": stage_info.get("file_path"),
                        "reused_converted_pdf": bool(
                            stage_info.get("reused_converted_pdf", False)
                        ),
                        "converted_pdf": stage_info.get("converted_pdf"),
                        "converted_pdf_candidates": stage_info.get(
                            "converted_pdf_candidates", []
                        ),
                        "error": str(exc),
                    }
                )
        if enqueued:
            LOGGER.info("Processing enqueue queue: %d document(s).", len(enqueued))
        else:
            LOGGER.warning(
                "No new documents were enqueued; processing queue anyway to resume "
                "existing pending/failed documents."
            )
        await rag.apipeline_process_enqueue_documents()
        LOGGER.info("Processing queue finished.")
        LOGGER.info("Collecting document status.")
        documents = await _collect_doc_status(rag)
    finally:
        LOGGER.info("Finalizing LightRAG storages.")
        await rag.finalize_storages()

    return {
        "track_id": track_id,
        "elapsed_seconds": time.time() - start,
        "enqueued_count": len(enqueued),
        "skipped_existing_count": len(skipped_existing),
        "failed_enqueue_count": len(failures),
        "enqueued": enqueued,
        "skipped_existing": skipped_existing,
        "failures": failures,
        "reuse": _reuse_summary(enqueued + skipped_existing + failures, config),
        "documents": documents,
    }


async def _run_query(config: BuildConfig, query_text: str) -> dict[str, Any]:
    from lightrag.base import QueryParam

    rag = _make_rag(config, include_reranker=config.enable_query_rerank)
    param_kwargs: dict[str, Any] = {
        "mode": config.query_mode,
        "stream": False,
        "enable_rerank": config.enable_query_rerank,
    }
    if config.top_k is not None:
        param_kwargs["top_k"] = config.top_k
    if config.chunk_top_k is not None:
        param_kwargs["chunk_top_k"] = config.chunk_top_k
    param = QueryParam(**param_kwargs)
    start = time.time()
    try:
        LOGGER.info("Initializing LightRAG storages for query.")
        await rag.initialize_storages()
        LOGGER.info(
            "Query start: mode=%s top_k=%s chunk_top_k=%s rerank=%s",
            param.mode,
            param.top_k,
            param.chunk_top_k,
            param.enable_rerank,
        )
        response = await rag.aquery(query_text, param=param)
        LOGGER.info("Query finished in %.2f seconds.", time.time() - start)
    finally:
        LOGGER.info("Finalizing LightRAG storages after query.")
        await rag.finalize_storages()

    return {
        "query": query_text,
        "response": response,
        "elapsed_seconds": time.time() - start,
        "query_param": {
            "mode": param.mode,
            "top_k": param.top_k,
            "chunk_top_k": param.chunk_top_k,
            "enable_rerank": param.enable_rerank,
            "max_entity_tokens": param.max_entity_tokens,
            "max_relation_tokens": param.max_relation_tokens,
            "max_total_tokens": param.max_total_tokens,
        },
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
    parser.add_argument("--enable-build-rerank", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--query-file", default=None)
    parser.add_argument("--query-only", action="store_true")
    parser.add_argument(
        "--query-mode",
        default=os.getenv("LIGHTRAG_QUERY_MODE", "mix"),
        choices=("local", "global", "hybrid", "naive", "mix", "bypass"),
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--chunk-top-k", type=int, default=None)
    query_rerank_group = parser.add_mutually_exclusive_group()
    query_rerank_group.add_argument(
        "--enable-query-rerank",
        dest="enable_query_rerank",
        action="store_true",
        default=True,
    )
    query_rerank_group.add_argument(
        "--disable-query-rerank",
        dest="enable_query_rerank",
        action="store_false",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_env_file(Path(args.env_file).expanduser())
    config = _build_config(args)
    _apply_runtime_env(config)
    log_file = _setup_logging(config.report_dir)
    LOGGER.info("Log file: %s", log_file)

    if not config.query_only and not config.raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {config.raw_dir}")
    files = [] if config.query_only else _scan_files(config.raw_dir, config.extensions, config.recursive)
    if config.max_files is not None:
        files = files[: max(0, int(config.max_files))]
    _log_settings(config, files)
    _validate_runtime_config()
    file_records = _build_file_reuse_preview(config, files)
    reuse = _reuse_summary(file_records, config)
    LOGGER.info(
        "LibreOffice PDF reuse preview: reused=%d fallback=%d total=%d",
        reuse["reused_converted_pdf_count"],
        reuse["fallback_to_original_count"],
        reuse["file_count"],
    )

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
        "query_only": config.query_only,
        "query": config.query,
        "query_file": config.query_file.as_posix() if config.query_file else None,
        "query_mode": config.query_mode,
        "top_k": config.top_k,
        "chunk_top_k": config.chunk_top_k,
        "enable_build_rerank": config.enable_build_rerank,
        "enable_query_rerank": config.enable_query_rerank,
        "reuse": reuse,
        "file_records": file_records,
        "settings": _settings_summary(config, files),
    }

    if config.dry_run:
        _write_json(config.report_dir / "build_summary.json", summary_base)
        _write_json(config.report_dir / "summary.json", _build_compact_summary(summary_base))
        LOGGER.info("Dry run complete: %d files", len(files))
        return 0

    result: dict[str, Any] = {}
    exit_code = 0
    if not config.query_only:
        result = asyncio.run(_run_build(config, files))
    if config.query:
        query_result = asyncio.run(_run_query(config, config.query))
        result["query_result"] = query_result
        _write_json(config.report_dir / "query_response.json", query_result)
        print(query_result["response"])
    elif config.query_only:
        raise ValueError("--query-only requires --query or --query-file.")

    summary = {**summary_base, **result}
    compact_summary = _build_compact_summary(summary_base, result)
    if not config.query_only:
        exit_code = 1 if compact_summary.get("failed_count") else 0
    _write_json(config.report_dir / "build_summary.json", summary)
    _write_json(config.report_dir / "summary.json", compact_summary)
    _write_json(config.report_dir / "failed_files.json", compact_summary.get("failed_files", []))
    _write_json(config.report_dir / "documents_status.json", result.get("documents", {}))
    LOGGER.info("Build finished. Summary: %s", config.report_dir / "summary.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
