from __future__ import annotations

import asyncio
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
        "LIGHTRAG_VECTOR_STORAGE",
        "QDRANT_ENABLE_SPARSE_BM25",
        "QDRANT_SPARSE_BM25_MODEL",
        "QDRANT_RETRIEVAL_MODE",
        "QDRANT_COLLECTION_PREFIX",
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
    assert adapter.os.environ["LIGHTRAG_VECTOR_STORAGE"] == "QdrantHybridBM25VectorDBStorage"
    assert adapter.os.environ["QDRANT_ENABLE_SPARSE_BM25"] == "true"
    assert adapter.os.environ["QDRANT_SPARSE_BM25_MODEL"] == "Qdrant/bm25"
    assert adapter.os.environ["QDRANT_RETRIEVAL_MODE"] == "hybrid"
    assert adapter.os.environ["QDRANT_COLLECTION_PREFIX"] == "local_lightrag_bm25"


def test_register_local_hybrid_bm25_storage(monkeypatch):
    adapter = load_adapter()
    from lightrag import kg

    kg.STORAGES.pop("QdrantHybridBM25VectorDBStorage", None)
    kg.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"] = [
        item
        for item in kg.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
        if item != "QdrantHybridBM25VectorDBStorage"
    ]
    kg.STORAGE_ENV_REQUIREMENTS.pop("QdrantHybridBM25VectorDBStorage", None)

    adapter._register_local_hybrid_bm25_storage()

    assert (
        kg.STORAGES["QdrantHybridBM25VectorDBStorage"]
        == "local_lightrag.qdrant_hybrid_bm25"
    )
    assert (
        "QdrantHybridBM25VectorDBStorage"
        in kg.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
    )
    assert kg.STORAGE_ENV_REQUIREMENTS["QdrantHybridBM25VectorDBStorage"] == [
        "QDRANT_URL"
    ]


def test_hybrid_bm25_collection_name_includes_prefix_and_workspace(monkeypatch):
    from lightrag.utils import EmbeddingFunc
    from local_lightrag.qdrant_hybrid_bm25 import QdrantHybridBM25VectorDBStorage

    monkeypatch.setenv("QDRANT_COLLECTION_PREFIX", "local_lightrag_bm25")
    monkeypatch.setenv("QDRANT_ENABLE_SPARSE_BM25", "true")
    monkeypatch.setenv("QDRANT_RETRIEVAL_MODE", "hybrid")

    async def embed(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    storage = QdrantHybridBM25VectorDBStorage(
        namespace="chunks",
        workspace="internal_lightrag",
        global_config={
            "embedding_batch_num": 4,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=128,
            func=embed,
            model_name="bge-m3",
        ),
        meta_fields={"content"},
    )

    assert storage.final_namespace == "local_lightrag_bm25_internal_lightrag_vdb_chunks_bge_m3_4d"
    assert storage.retrieval_mode == "hybrid"
    assert storage.enable_sparse_bm25 is True


def test_hybrid_bm25_get_vectors_by_ids_returns_dense_vector(monkeypatch):
    from lightrag.utils import EmbeddingFunc
    from local_lightrag.qdrant_hybrid_bm25 import QdrantHybridBM25VectorDBStorage

    async def embed(texts, **kwargs):
        return [[0.0] * 4 for _ in texts]

    storage = QdrantHybridBM25VectorDBStorage(
        namespace="chunks",
        workspace="internal_lightrag",
        global_config={
            "embedding_batch_num": 4,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=4,
            max_token_size=128,
            func=embed,
            model_name="bge-m3",
        ),
        meta_fields={"content"},
    )

    class Point:
        payload = {"id": "chunk-1"}
        vector = {"dense": [0.1, 0.2, 0.3, 0.4], "bm25": object()}

    class Client:
        def retrieve(self, **kwargs):
            return [Point()]

    storage._client = Client()

    vectors = asyncio.run(storage.get_vectors_by_ids(["chunk-1"]))

    assert vectors == {"chunk-1": [0.1, 0.2, 0.3, 0.4]}


def test_compact_summary_counts_processed_failed_and_enqueue_errors(tmp_path):
    adapter = load_adapter()
    base = {
        "workspace": "ws",
        "raw_dir": str(tmp_path / "raw"),
        "working_dir": str(tmp_path / "work"),
        "input_dir": str(tmp_path / "inputs"),
        "report_dir": str(tmp_path / "reports"),
        "file_count": 3,
        "files": ["good.pdf", "bad.pdf", "enqueue.docx"],
        "settings": {"env": {"MAX_PARALLEL_INSERT": "2"}},
    }
    result = {
        "elapsed_seconds": 12.5,
        "enqueued_count": 2,
        "skipped_existing_count": 1,
        "failed_enqueue_count": 1,
        "failures": [{"source": "enqueue.docx", "error": "cannot stage"}],
        "documents": {
            "documents": [
                {
                    "doc_id": "doc-good",
                    "payload": {
                        "file_path": "good.pdf",
                        "status": "processed",
                        "track_id": "track-a",
                    },
                },
                {
                    "doc_id": "doc-bad",
                    "payload": {
                        "file_path": "bad.pdf",
                        "status": "failed",
                        "error_msg": "VLM prompt too long",
                    },
                },
            ]
        },
    }

    summary = adapter._build_compact_summary(base, result)

    assert summary["succeeded_count"] == 1
    assert summary["failed_count"] == 2
    assert summary["status_counts"] == {"failed": 1, "processed": 1}
    assert summary["failed_files"][0]["file"] == "enqueue.docx"
    assert summary["failed_files"][1]["file"] == "bad.pdf"


def test_existing_processed_or_failed_sources_are_not_enqueued_again(tmp_path):
    adapter = load_adapter()

    known = {
        "already.pdf": {"status": "processed", "doc_id": "doc-1"},
        "retry.pdf": {"status": "failed", "doc_id": "doc-2"},
    }

    assert adapter._existing_doc_record_for_source(tmp_path / "already.pdf", known)[
        "status"
    ] == "processed"
    assert adapter._existing_doc_record_for_source(tmp_path / "retry.pdf", known)[
        "status"
    ] == "failed"
    assert adapter._existing_doc_record_for_source(tmp_path / "new.pdf", known) is None


def test_resolve_query_text_accepts_inline_or_file(tmp_path):
    adapter = load_adapter()

    inline_args = adapter.build_parser().parse_args(["--query", "  what is indexed?  "])
    assert adapter._resolve_query_text(inline_args) == "what is indexed?"

    query_file = tmp_path / "question.txt"
    query_file.write_text("\n请总结这批文档\n", encoding="utf-8")
    file_args = adapter.build_parser().parse_args(["--query-file", str(query_file)])
    assert adapter._resolve_query_text(file_args) == "请总结这批文档"
