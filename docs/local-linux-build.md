# Local Linux Build With Official LightRAG

This repository keeps official LightRAG source unchanged. Local adaptation lives
in `configs/local-qwen3vl.env.example`, `scripts/build_internal_lightrag.py`,
and this document.

## What Is Preserved From RAG_LUND

- LLM/VLM endpoint: `http://localhost:8001/v1`
- LLM/VLM model: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`
- Embedding model loaded in-process: `/data/h50056787/models/bge-m3`
- Reranker loaded in-process: `/data/h50056787/models/bge-reranker-v2-m3`
- Neo4j graph storage and Qdrant vector storage
- `MAX_SOURCE_IDS_PER_ENTITY=99999`
- `MAX_SOURCE_IDS_PER_RELATION=99999`
- Conservative document pipeline concurrency: `MAX_PARALLEL_INSERT=2`

The embedding and reranker do not use HTTP ports in this build script. They are
loaded with `SentenceTransformer` and `CrossEncoder` inside the Python process.

## What Is Not Migrated

This is intentionally an official LightRAG layout. It does not migrate RAG_LUND
custom source changes:

- entity disambiguation
- synonym edges
- strict relation endpoint matching
- custom prompt modifications
- Qdrant dense+sparse BM25 hybrid collections

Official Qdrant storage in this repository is dense-vector only.

## Install

```bash
cd /data/y50056788/Yaliang/projects/LightRAG-Qwen3VL-Local
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[api]"
pip install sentence-transformers
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
- `build_summary.json`
- `failed_files.json`
- `documents_status.json`

## Query Smoke Test

After a build, use the official LightRAG API or a short SDK script against the
same `WORKSPACE`, `WORKING_DIR`, Neo4j, and Qdrant settings. Use rerank only
when the local CrossEncoder can fit on the selected GPU.

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
