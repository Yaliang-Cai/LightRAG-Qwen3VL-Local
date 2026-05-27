# Local Linux Build With Official LightRAG

This repository keeps official LightRAG source unchanged. Local adaptation lives
in `configs/local-qwen3vl.env.example`, `scripts/build_internal_lightrag.py`,
and this document.

## What Is Preserved From RAG_LUND

- LLM/VLM endpoint: `http://localhost:8001/v1`
- LLM/VLM model: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`
- Embedding model loaded in-process: `/data/h50056787/models/bge-m3`
- Reranker loaded in-process: `/data/h50056787/models/bge-reranker-v2-m3`
- Neo4j graph storage and local Qdrant dense+BM25 vector storage
- `MAX_SOURCE_IDS_PER_ENTITY=99999`
- `MAX_SOURCE_IDS_PER_RELATION=99999`
- Conservative document pipeline concurrency: `MAX_PARALLEL_INSERT=2`
- Build-time reranker disabled by default. Query-time reranker enabled by
  default.

The embedding and reranker do not use HTTP ports in this build script. They are
loaded with `SentenceTransformer` and `CrossEncoder` inside the Python process.
The CrossEncoder is only attached for query runs unless you explicitly pass
`--enable-build-rerank`.

## What Is Not Migrated

This is intentionally an official LightRAG layout. It does not migrate RAG_LUND
custom source changes:

- entity disambiguation
- synonym edges
- strict relation endpoint matching
- custom prompt modifications
- RAG_LUND's full retrieval router, PPR profile, and evaluation-only hybrid
  profiles

Official LightRAG's built-in Qdrant storage is dense-vector only. This
repository keeps official `lightrag/` source unchanged and adds a local adapter
named `QdrantHybridBM25VectorDBStorage` under `local_lightrag/` for the build
script. The adapter writes dense vectors plus Qdrant BM25 sparse vectors and
queries them with RRF fusion by default.

## Install

```bash
cd /data/y50056788/Yaliang/projects/LightRAG-Qwen3VL-Local
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[api]"
pip install sentence-transformers fastembed
cp configs/local-qwen3vl.env.example configs/local-qwen3vl.env
```

Edit `configs/local-qwen3vl.env` and set:

- `NEO4J_PASSWORD`
- model paths if they differ from the RAG_LUND machine
- `MINERU_LOCAL_ENDPOINT` if your MinerU service is not on port `8000`

## Start Services

Start Neo4j and Qdrant with your normal local deployment. The defaults expected
by `configs/local-qwen3vl.env` are:

```bash
export NEO4J_URI=bolt://localhost:7687
export QDRANT_URL=http://localhost:6333
```

The local hybrid adapter uses separate Qdrant collection names from both the
official dense-only profile and RAG_LUND. With the default workspace, collection
names are shaped like:

```text
local_lightrag_bm25_internal_lightrag_vdb_chunks_bge_m3_1024d
local_lightrag_bm25_internal_lightrag_vdb_entities_bge_m3_1024d
local_lightrag_bm25_internal_lightrag_vdb_relationships_bge_m3_1024d
```

Payloads still include `workspace_id=internal_lightrag`. Neo4j remains isolated
by the `:internal_lightrag` label. Do not set `QDRANT_WORKSPACE` or
`NEO4J_WORKSPACE` to a RAG_LUND workspace such as `internal` unless you
intentionally want to share logical workspace ids.

Start Qwen3-VL with the same OpenAI-compatible endpoint used by RAG_LUND:

```bash
vllm serve /data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key EMPTY
```

For PDF, Office, and image files, start the official LightRAG MinerU-compatible
local service configured by:

```bash
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://127.0.0.1:8000
MINERU_LOCAL_BACKEND=hybrid-auto-engine
```

Text and Markdown files do not need MinerU.

## Build

Dry run:

```bash
source .venv/bin/activate
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --raw-dir /data/y50056788/Yaliang/datasets_raw \
  --max-files 2 \
  --dry-run
```

Full build:

```bash
source .venv/bin/activate
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --raw-dir /data/y50056788/Yaliang/datasets_raw \
  --workspace internal_lightrag \
  --max-concurrent-files 2
```

Reports are written under:

```text
/data/y50056788/Yaliang/internal_lightrag/reports/<timestamp>/
```

Important report files:

- `build_internal_lightrag.log`
- `summary.json`: compact RAG_LUND-style run summary with success/failure
  counts, status counts, failed files, and selected document statuses.
- `build_summary.json`
- `failed_files.json`
- `documents_status.json`

The script stages files into `INPUT_DIR/<workspace>/` with stable hashed names.
Before enqueueing, it checks LightRAG `doc_status` by that staged basename. If a
document already exists, it is not enqueued again; the script still calls
`apipeline_process_enqueue_documents()` so official LightRAG can resume
`pending`, `parsing`, `analyzing`, `processing`, and `failed` records. Fully
processed documents are therefore skipped on rerun. Failed documents are retried
by the official pipeline. MinerU is not rerun for processed documents; a failed
document may invoke MinerU again if the previous failure happened before parse
artifacts/full document content were completed or those artifacts are missing.

## Query Smoke Test

Put a question in a file:

```bash
cat >/tmp/lightrag_question.txt <<'EOF'
请总结这批文档里的核心实体和关系。
EOF
```

Then query the built workspace:

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-file /tmp/lightrag_question.txt \
  --query-only \
  --top-k 40 \
  --chunk-top-k 20
```

You can also query inline:

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query "请回答这个库里 Lund 相关项目的主要内容。" \
  --query-only
```

Query rerank is on by default and loads the local
`/data/h50056787/models/bge-reranker-v2-m3` CrossEncoder in the query process.
If GPU memory is tight, add `--disable-query-rerank`. The build path still does
not attach or load the reranker.

Do not edit official `lightrag/constants.py` just to tune retrieval. Use env or
CLI:

- `TOP_K` / `--top-k`
- `CHUNK_TOP_K` / `--chunk-top-k`
- `MAX_ENTITY_TOKENS`, `MAX_RELATION_TOKENS`, `MAX_TOTAL_TOKENS`

The local Qdrant adapter indexes both dense and BM25 sparse vectors. Query mode
is controlled by `QDRANT_RETRIEVAL_MODE`:

- `hybrid`: dense and BM25 prefetch, fused by Qdrant RRF. This is the default.
- `dense`: dense vector search only.
- `bm25`: sparse lexical search only.

Official LightRAG reranks retrieved candidates, not the whole database. For
vector chunks, `_get_vector_context` first queries Qdrant with
`search_top_k = chunk_top_k or top_k`. The local adapter applies
`QDRANT_RETRIEVAL_MODE` inside that Qdrant query. Then
`process_chunks_unified` sends the deduplicated candidate list to the reranker
with `top_n = chunk_top_k or len(unique_chunks)`.

## Official LightRAG Concurrency

LightRAG concurrency is layered:

- `MAX_PARALLEL_INSERT`: document/file pipeline concurrency. Official default is `2`.
- `MAX_ASYNC`: LLM call concurrency. Official default is `4`.
- `EMBEDDING_FUNC_MAX_ASYNC`: embedding call concurrency. Official default is `8`.
- `EMBEDDING_BATCH_NUM`: texts per embedding batch. Official default is `10`.
- `MAX_ASYNC_RERANK`: rerank concurrency. When unset, it falls back to `MAX_ASYNC`.
- `MAX_PARALLEL_PARSE_NATIVE`, `MAX_PARALLEL_PARSE_MINERU`,
  `MAX_PARALLEL_PARSE_DOCLING`: parser worker concurrency.
- `MAX_PARALLEL_ANALYZE`: multimodal analysis worker concurrency.

There is no RAG_LUND-style `RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM` switch in
official LightRAG. Multimodal load is controlled by `VLM_PROCESS_ENABLE`,
`VLM_MAX_ASYNC_LLM`, parser/analyze worker settings, and base LLM concurrency.

This local profile sets:

- `MAX_PARALLEL_INSERT=2`
- `MAX_ASYNC=16`, matching the RAG_LUND LLM async default rather than official
  LightRAG's `4`
- `EMBEDDING_FUNC_MAX_ASYNC=4`
- `EMBEDDING_BATCH_NUM=4`
- `MAX_ASYNC_RERANK=1`
- `VLM_MAX_ASYNC_LLM=2`
- `MAX_PARALLEL_PARSE_NATIVE=2`
- `MAX_PARALLEL_PARSE_MINERU=1`
- `MAX_PARALLEL_PARSE_DOCLING=1`
- `MAX_PARALLEL_ANALYZE=2`

For local Qwen3-VL, keep MinerU/Docling parse at `1` until you have measured
CPU/GPU memory. Multimodal analyze and VLM calls default to `2`; reduce both to
`1` if vLLM memory or latency becomes unstable. Native text parse can stay at
`2` because it is much lighter.

## Multimodal Token Guard

RAG_LUND previously hit multimodal chunks above `65536` tokens. This adapter
does not patch official LightRAG source; it uses official guards:

- `MAX_EXTRACT_INPUT_TOKENS=20480`
- `EMBEDDING_TOKEN_LIMIT=8192`

Official `pipeline.py` trims multimodal analysis/extraction input against
`MAX_EXTRACT_INPUT_TOKENS` and errors if a single frame or sidecar cannot fit.
If a document still exceeds the local model context, lower
`MAX_EXTRACT_INPUT_TOKENS`, reduce parser options in `LIGHTRAG_PARSER` (for
example fewer image/table/equation analyses), or process that document class
separately. Raising the value is possible only if your served Qwen3-VL context
actually supports it.
