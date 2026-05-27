from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_internal_lightrag.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("build_internal_lightrag", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_defaults_keep_reranker_off_for_build_but_on_for_query():
    adapter = load_adapter()

    args = adapter.build_parser().parse_args([])

    assert args.enable_build_rerank is False
    assert args.enable_query_rerank is True
    assert args.query is None
    assert args.query_file is None
    assert args.query_only is False

    disabled_args = adapter.build_parser().parse_args(["--disable-query-rerank"])
    assert disabled_args.enable_query_rerank is False


def test_runtime_env_defaults_match_local_concurrency(monkeypatch, tmp_path):
    adapter = load_adapter()
    for key in (
        "MAX_ASYNC",
        "MAX_PARALLEL_INSERT",
        "MAX_PARALLEL_PARSE_NATIVE",
        "MAX_PARALLEL_PARSE_MINERU",
        "MAX_PARALLEL_PARSE_DOCLING",
        "MAX_PARALLEL_ANALYZE",
        "EMBEDDING_BATCH_NUM",
        "RAGANYTHING_EMBEDDING_BATCH_NUM",
        "MAX_EXTRACT_INPUT_TOKENS",
        "EMBEDDING_TOKEN_LIMIT",
        "RERANK_BY_DEFAULT",
        "VLM_MAX_ASYNC_LLM",
    ):
        monkeypatch.delenv(key, raising=False)

    config = adapter.BuildConfig(
        raw_dir=tmp_path / "raw",
        storage_root=tmp_path / "storage",
        working_dir=tmp_path / "working",
        input_dir=tmp_path / "inputs",
        report_dir=tmp_path / "reports",
        workspace="test_workspace",
        max_files=None,
        max_parallel_insert=2,
        recursive=True,
        extensions=(".txt",),
        dry_run=True,
        enable_build_rerank=False,
        enable_query_rerank=True,
        query=None,
        query_file=None,
        query_only=False,
        query_mode="mix",
        top_k=None,
        chunk_top_k=None,
    )

    adapter._apply_runtime_env(config)

    assert adapter.os.environ["MAX_ASYNC"] == "16"
    assert adapter.os.environ["MAX_PARALLEL_INSERT"] == "2"
    assert adapter.os.environ["MAX_PARALLEL_PARSE_NATIVE"] == "2"
    assert adapter.os.environ["MAX_PARALLEL_PARSE_MINERU"] == "1"
    assert adapter.os.environ["MAX_PARALLEL_PARSE_DOCLING"] == "1"
    assert adapter.os.environ["MAX_PARALLEL_ANALYZE"] == "2"
    assert adapter.os.environ["EMBEDDING_BATCH_NUM"] == "4"
    assert adapter.os.environ["RAGANYTHING_EMBEDDING_BATCH_NUM"] == "4"
    assert adapter.os.environ["MAX_EXTRACT_INPUT_TOKENS"] == "20480"
    assert adapter.os.environ["EMBEDDING_TOKEN_LIMIT"] == "8192"
    assert adapter.os.environ["RERANK_BY_DEFAULT"] == "true"
    assert adapter.os.environ["VLM_MAX_ASYNC_LLM"] == "2"


def test_resolve_query_text_accepts_inline_or_file(tmp_path):
    adapter = load_adapter()

    inline_args = adapter.build_parser().parse_args(["--query", "  what is indexed?  "])
    assert adapter._resolve_query_text(inline_args) == "what is indexed?"

    query_file = tmp_path / "question.txt"
    query_file.write_text("\n请总结这批文档\n", encoding="utf-8")
    file_args = adapter.build_parser().parse_args(["--query-file", str(query_file)])
    assert adapter._resolve_query_text(file_args) == "请总结这批文档"
