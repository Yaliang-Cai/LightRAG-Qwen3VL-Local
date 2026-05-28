# 原生 LightRAG 本地建库说明

这个仓库保持官方 `lightrag/` 源码不改。本地适配只放在
`configs/local-qwen3vl.env.example`、`scripts/build_internal_lightrag.py`、
`local_lightrag/` 和本文档里。

## 本地默认配置

- 输入目录：`/data/y50056788/Yaliang/datasets_raw`
- 原生 LightRAG 输出目录：`/data/y50056788/Yaliang/internal_lightrag/`
- workspace：`internal_lightrag`
- LLM/VLM endpoint：`http://localhost:8001/v1`
- LLM/VLM model：`Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`
- Tokenizer：本地加载 `/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8`
- Embedding：进程内加载 `/data/h50056787/models/bge-m3`，不开端口
- Reranker：进程内加载 `/data/h50056787/models/bge-reranker-v2-m3`，不开端口
- 建库默认不加载 reranker；查询默认启用 reranker
- Neo4j graph storage：`Neo4JStorage`
- Qdrant vector storage：本地 adapter `QdrantHybridBM25VectorDBStorage`
- source id 上限：`MAX_SOURCE_IDS_PER_ENTITY=999999`，`MAX_SOURCE_IDS_PER_RELATION=999999`

这不是 RAG_LUND 开发版迁移。以下开发版自定义逻辑不会带过来：

- entity disambiguation
- 同义边
- strict endpoint match
- 自定义 prompt 改动
- RAG_LUND 的完整 retrieval router / PPR profile / evaluation-only hybrid profile

## 和开发版的隔离

默认不要改这些值：

```env
WORKSPACE=internal_lightrag
WORKING_DIR=/data/y50056788/Yaliang/internal_lightrag/rag_workspace
INPUT_DIR=/data/y50056788/Yaliang/internal_lightrag/inputs
REPORT_DIR=/data/y50056788/Yaliang/internal_lightrag/reports
QDRANT_COLLECTION_PREFIX=local_lightrag_bm25
```

默认 Qdrant collection 名类似：

```text
local_lightrag_bm25_internal_lightrag_vdb_chunks_bge_m3_1024d
local_lightrag_bm25_internal_lightrag_vdb_entities_bge_m3_1024d
local_lightrag_bm25_internal_lightrag_vdb_relationships_bge_m3_1024d
```

Neo4j 也会用 `internal_lightrag` workspace/label 隔离。不要把
`QDRANT_WORKSPACE`、`NEO4J_WORKSPACE` 或 `WORKSPACE` 改成开发版的
`internal`，否则就不是隔离建库。

## 复用开发版 LibreOffice PDF

脚本默认只读复用开发版已经转换过的 PDF：

```env
REUSE_DEV_LIBREOFFICE_PDFS=true
DEV_LIBREOFFICE_PDF_ROOT=/data/y50056788/Yaliang/internal/output/internal
```

对 Office 文件，脚本会按开发版规则优先查：

```text
/data/y50056788/Yaliang/internal/output/internal/<safe_stem>/<source_stem>.pdf
/data/y50056788/Yaliang/internal/output/internal/<source_stem>.pdf
```

`safe_stem` 规则和开发版一致：保留字母数字以及 `._-`，其他字符替换成 `_`。
如果 PDF 存在且非空就复用；即使 PDF 比源文件旧，也只会记录 warning 后继续复用。
如果找不到 PDF，会回退到原 Office 文件，让官方 LightRAG/MinerU 链路继续处理。

复用不会写入 `/data/y50056788/Yaliang/internal/`。脚本只把复用到的 PDF stage 到：

```text
/data/y50056788/Yaliang/internal_lightrag/inputs/internal_lightrag/
```

## 安装环境

```bash
cd /data/y50056788/Yaliang/projects/LightRAG-Qwen3VL-Local
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[api]"
pip install sentence-transformers fastembed qdrant-client neo4j
cp configs/local-qwen3vl.env.example configs/local-qwen3vl.env
```

编辑 `configs/local-qwen3vl.env`，至少确认：

- `NEO4J_PASSWORD`
- `LLM_TOKENIZER_MODEL_PATH`
- embedding/reranker 模型路径
- `MINERU_LOCAL_ENDPOINT`
- `DEV_LIBREOFFICE_PDF_ROOT`

## 启动服务

Neo4j 和 Qdrant 使用本地服务即可，默认配置是：

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_DATABASE=neo4j
export QDRANT_URL=http://localhost:6333
```

Qwen3-VL 使用和 RAG_LUND 开发版一致的 OpenAI-compatible endpoint：

```bash
vllm serve /data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key EMPTY
```

脚本会通过 `LLM_TOKENIZER_MODEL_PATH` 注入本地 HuggingFace tokenizer，并设置
`local_files_only=True`。这样不会触发官方默认 `tiktoken` 下载
`o200k_base.tiktoken`，也不需要访问 `openaipublic.blob.core.windows.net`。
如果你的 Qwen3-VL 模型目录不同，改 `configs/local-qwen3vl.env`：

```env
LLM_TOKENIZER_MODEL_PATH=/你的/Qwen3-VL-30B-A3B-Instruct-FP8/本地目录
```

PDF、Office、图片文件需要 MinerU-compatible local service：

```env
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://127.0.0.1:8000
MINERU_LOCAL_BACKEND=hybrid-auto-engine
MINERU_LOCAL_PARSE_METHOD=auto
MINERU_LOCAL_IMAGE_ANALYSIS=true
```

`.txt` 和 `.md` 使用 legacy text ingestion，不需要 MinerU。
真实建库前脚本会检查 `MINERU_LOCAL_ENDPOINT/health`。这里必须是 `mineru-api`
或 `mineru-router` 的 REST API，不是 JSON-RPC/MCP 服务；正确服务需要提供
`/health`、`/tasks`、`/tasks/{task_id}` 和 `/tasks/{task_id}/result`。

LightRAG 官方 parser routing 使用无点后缀，不是 shell glob。例如：

```env
LIGHTRAG_PARSER=pdf:mineru-iteP,docx:mineru-iteP,txt:legacy-F,md:legacy-F
```

不要写成 `*.pdf:mineru-iteP`，否则官方 routing 匹配不到，会回退到 legacy parser。
当前官方 MinerU routing 不支持 `tif/tiff`，本地脚本默认不会扫描这两类文件；旧 env
里残留的 `tif/tiff:mineru-...` 规则会被脚本自动丢弃。

## 建库

先 dry-run 看文件数、配置和可复用 PDF 命中情况：

```bash
source .venv/bin/activate
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --dry-run \
  --max-files 10
```

正式建库可以直接运行：

```bash
source .venv/bin/activate
python scripts/build_internal_lightrag.py
```

等价的显式命令：

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --raw-dir /data/y50056788/Yaliang/datasets_raw \
  --workspace internal_lightrag \
  --max-concurrent-files 2
```

报告写到：

```text
/data/y50056788/Yaliang/internal_lightrag/reports/<timestamp>/
```

重要文件：

- `build_internal_lightrag.log`：详细日志
- `summary.json`：紧凑 summary，包含成功/失败、失败文件、doc 状态、PDF 复用统计
- `build_summary.json`：完整运行记录
- `failed_files.json`：失败文件列表
- `documents_status.json`：LightRAG doc_status 快照

单个文件 enqueue 失败会记录到 summary 并继续处理后续文件。再次运行同一个命令时，
脚本会根据 staged basename 查 `doc_status`，已存在的文档不会重复 enqueue；仍会调用
`apipeline_process_enqueue_documents()`，让官方 pipeline 继续处理 pending/failed 状态。
已 processed 的文档不会重新跑 MinerU。failed 文档是否重新跑 MinerU 取决于官方
LightRAG 上一次失败时是否已经留下可复用的 parse/full-doc artifact。

## 查询

建库后可以直接问问题：

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-only \
  --query "请总结这个库里的核心实体和关系。"
```

也可以把问题放到文件里：

```bash
cat >/tmp/lightrag_question.txt <<'EOF'
请总结这批文档里的主要项目、实体和关系。
EOF

python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-only \
  --query-file /tmp/lightrag_question.txt \
  --top-k 40 \
  --chunk-top-k 20
```

查询默认启用 rerank，会在查询进程内加载
`/data/h50056787/models/bge-reranker-v2-m3`。显存紧张时加：

```bash
--disable-query-rerank
```

建库默认不加载 reranker，除非显式传：

```bash
--enable-build-rerank
```

## Dense + BM25

官方 LightRAG 的 Qdrant storage 本身是 dense-only。这个仓库不改官方源码，而是在
`local_lightrag/` 下注册 `QdrantHybridBM25VectorDBStorage`，写入 dense vector 和
Qdrant BM25 sparse vector。

检索模式由 `QDRANT_RETRIEVAL_MODE` 控制：

- `hybrid`：dense + BM25 prefetch 后用 Qdrant RRF fusion，默认值
- `dense`：只走 dense vector
- `bm25`：只走 sparse lexical search

官方 LightRAG rerank 的对象不是全库。对 chunk 来说，LightRAG 先用
`search_top_k = chunk_top_k or top_k` 从 Qdrant 取候选 chunk；本地 adapter 在这一步执行
dense/BM25/hybrid；然后 `process_chunks_unified` 对去重后的候选列表 rerank，再取
`chunk_top_k`。

## 并发

本地默认：

- `MAX_PARALLEL_INSERT=2`
- `MAX_ASYNC=16`
- `EMBEDDING_FUNC_MAX_ASYNC=4`
- `EMBEDDING_BATCH_NUM=4`
- `MAX_ASYNC_RERANK=1`
- `MAX_PARALLEL_PARSE_NATIVE=2`
- `MAX_PARALLEL_PARSE_MINERU=1`
- `MAX_PARALLEL_PARSE_DOCLING=1`
- `MAX_PARALLEL_ANALYZE=2`
- `VLM_MAX_ASYNC_LLM=2`

LightRAG 并发是分层的：

- `MAX_PARALLEL_INSERT`：文档/文件 pipeline 并发
- `MAX_ASYNC`：LLM 调用并发
- `EMBEDDING_FUNC_MAX_ASYNC`：embedding 调用并发
- `EMBEDDING_BATCH_NUM`：单次 embedding batch
- `MAX_ASYNC_RERANK`：rerank 并发
- `MAX_PARALLEL_PARSE_NATIVE/MINERU/DOCLING`：parse worker 并发
- `MAX_PARALLEL_ANALYZE`：多模态 analyze worker 并发
- `VLM_MAX_ASYNC_LLM`：多模态 description/analyze 里 VLM role 的 LLM 并发

官方没有 RAG_LUND 那种单独的 `RAGANYTHING_MULTIMODAL_ITEM_PARALLELISM`。
多模态压力主要由 `VLM_PROCESS_ENABLE`、`VLM_MAX_ASYNC_LLM`、parse/analyze worker
和基础 `MAX_ASYNC` 共同控制。Qwen3-VL 显存不稳时，优先把
`MAX_PARALLEL_ANALYZE` 和 `VLM_MAX_ASYNC_LLM` 降到 `1`。

## 多模态 token 限制

开发版曾遇到 table/image description 输入超过本地模型上下文的问题。这个仓库不 patch
官方源码，默认用官方 guard：

```env
MAX_EXTRACT_INPUT_TOKENS=20480
EMBEDDING_TOKEN_LIMIT=8192
```

含义是：官方 pipeline 会尝试按 `MAX_EXTRACT_INPUT_TOKENS` 裁剪多模态 analyze/extract
输入。如果 table/image 本身或 prompt frame 裁完仍超过本地 Qwen3-VL 上下文，官方链路仍可能把
该文档标为 failed。遇到这种文件时，优先降低 `MAX_EXTRACT_INPUT_TOKENS`、减少
`LIGHTRAG_PARSER` 中 table/image/equation 分析，或把该类文件单独处理。
