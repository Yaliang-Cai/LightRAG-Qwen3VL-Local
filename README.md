# LightRAG-Qwen3VL-Local 本地交接说明

这个仓库是基于 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) 的本地适配 fork，用于在 Linux 机器上运行原生 LightRAG 建库和查询链路。

本仓库的目标是：使用原生 LightRAG 结构，接入本地 Qwen3-VL、进程内 embedding/reranker、Neo4j、Qdrant dense+BM25，并隔离输出到新的 `internal_lightrag` workspace。

更详细的 Linux 操作说明见 [docs/local-linux-build.md](docs/local-linux-build.md)。

## 1. 仓库用途

本仓库用于在 Linux 上运行：

- 原生 LightRAG 文件处理、建库、KG extraction、查询流程
- Qwen3-VL OpenAI-compatible LLM/VLM endpoint
- 本地 `SentenceTransformer` embedding
- 本地 `CrossEncoder` reranker
- Neo4j graph storage
- Qdrant dense+BM25 hybrid vector storage
- 开发版 LibreOffice 转换 PDF 的只读复用

默认路径和服务：

| 项目 | 默认值 |
| --- | --- |
| 输入目录 | `/data/y50056788/Yaliang/datasets_raw` |
| 输出目录 | `/data/y50056788/Yaliang/internal_lightrag/` |
| workspace | `internal_lightrag` |
| Qwen3-VL endpoint | `http://localhost:8001/v1` |
| Qwen3-VL model | `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` |
| Tokenizer | `/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8` |
| Embedding | `/data/h50056787/models/bge-m3` |
| Reranker | `/data/h50056787/models/bge-reranker-v2-m3` |
| MinerU | `http://127.0.0.1:8000` |
| Neo4j | `bolt://localhost:7687` |
| Qdrant | `http://localhost:6333` |

这不是 RAG_LUND 开发版迁移。以下开发版逻辑不会带过来：

- entity disambiguation
- 同义边
- strict endpoint match
- RAG_LUND retrieval router
- RAG_LUND PPR profile / evaluation-only hybrid profile
- 开发版自定义 prompt

开发版 LibreOffice PDF 只做只读复用：脚本会优先使用开发版已经转换好的 PDF 作为 LightRAG 输入，但 MinerU/LightRAG 原生解析链路仍会运行。

## 2. 快速开始

在 Linux 仓库目录执行：

```bash
cd /data/y50056788/Yaliang/LightRAG-0528
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[api]"
pip install sentence-transformers fastembed qdrant-client neo4j
cp configs/local-qwen3vl.env.example configs/local-qwen3vl.env
```

编辑配置：

```bash
nano configs/local-qwen3vl.env
```

至少确认这些值：

- `NEO4J_PASSWORD`
- `LLM_TOKENIZER_MODEL_PATH`
- `RAGANYTHING_EMBEDDING_MODEL_PATH`
- `RAGANYTHING_RERANK_MODEL_PATH`
- `MINERU_LOCAL_ENDPOINT`
- `DEV_LIBREOFFICE_PDF_ROOT`

`configs/local-qwen3vl.env` 不要提交真实密码或 token。

## 3. 必要服务

### Qwen3-VL vLLM

Qwen3-VL 默认使用和 RAG_LUND 开发版一致的 OpenAI-compatible endpoint：

```bash
vllm serve /data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key EMPTY
```

脚本会通过 `LLM_TOKENIZER_MODEL_PATH` 本地加载 HuggingFace tokenizer，避免 LightRAG 默认 tiktoken 下载 `o200k_base.tiktoken`。

### Neo4j 和 Qdrant

默认使用本地服务：

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_DATABASE=neo4j
export QDRANT_URL=http://localhost:6333
```

`NEO4J_PASSWORD` 建议写在 `configs/local-qwen3vl.env`，不要写进 README 或提交到远端。

### MinerU API

PDF、Office、图片文件需要 MinerU-compatible REST API：

```env
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://127.0.0.1:8000
MINERU_LOCAL_BACKEND=hybrid-auto-engine
MINERU_LOCAL_PARSE_METHOD=auto
MINERU_LOCAL_IMAGE_ANALYSIS=true
MINERU_POLL_INTERVAL_SECONDS=5
MINERU_MAX_POLLS=720
```

启动命令取决于当前 MinerU 安装方式，常见形式是：

```bash
mineru-api --host 127.0.0.1 --port 8000
```

如果命令不存在，检查当前 venv 是否安装 MinerU，或使用对应环境里的 MinerU 启动命令。LightRAG 侧要求这个服务提供 `/health`、`/tasks`、`/tasks/{task_id}` 和 `/tasks/{task_id}/result`。

`MINERU_LOCAL_IMAGE_ANALYSIS` 是 MinerU 内部 VLM 分析开关，不是 LightRAG 后续使用 Qwen3-VL 的多模态 description。显存紧张、MinerU transformers 崩溃、或只希望 MinerU 做解析时，可以改成：

```env
MINERU_LOCAL_IMAGE_ANALYSIS=false
```

LightRAG 后续是否调用 Qwen3-VL 做多模态 description 由 `VLM_PROCESS_ENABLE=true` 和 `VLM_MAX_ASYNC_LLM` 控制。

## 4. 建库

先 dry-run 检查配置、文件数和开发版 PDF 复用命中情况：

```bash
source .venv/bin/activate
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --dry-run \
  --max-files 10
```

正式建库：

```bash
source .venv/bin/activate
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env
```

等价的显式命令：

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --raw-dir /data/y50056788/Yaliang/datasets_raw \
  --workspace internal_lightrag \
  --max-concurrent-files 2
```

报告目录：

```text
/data/y50056788/Yaliang/internal_lightrag/reports/<timestamp>/
```

重要文件：

- `build_internal_lightrag.log`：详细日志
- `summary.json`：紧凑 summary，包含成功/失败、失败文件、doc 状态、PDF 复用统计
- `build_summary.json`：完整运行记录
- `failed_files.json`：失败文件列表
- `documents_status.json`：LightRAG doc_status 快照

单个文件 enqueue 或处理失败会记录到 summary，并继续处理后续文件。再次运行同一个命令时，脚本会根据 staged basename 检查 `doc_status`，已存在文档不会重复 enqueue；仍会调用官方 `apipeline_process_enqueue_documents()` 继续处理 pending/failed 状态。

已 processed 的文档不会重新跑 MinerU。failed 文档是否重新跑 MinerU，取决于官方 LightRAG 上次失败时是否已经留下可复用 parse/full-doc artifact。

## 5. 查询

建库后直接查询，不需要重新建库：

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-only \
  --query "你的问题"
```

指定 query mode、top_k 和 chunk_top_k：

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-only \
  --query "你的问题" \
  --query-mode mix \
  --top-k 40 \
  --chunk-top-k 20
```

也可以把问题放到文件里：

```bash
cat >/tmp/lightrag_question.txt <<'EOF'
请总结这批文档里的主要实体、关系和关键结论。
EOF

python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-only \
  --query-file /tmp/lightrag_question.txt
```

默认行为：

- `--query-only` 只表示只查询、不建库。
- 默认 `--query-mode mix`。
- `mix` 会使用 KG entity/relation 和 chunk 检索。
- 底层 Qdrant chunk retrieval 默认是 dense+BM25 hybrid，因为 `QDRANT_RETRIEVAL_MODE=hybrid`。
- 查询默认启用 reranker，因为 `RERANK_BY_DEFAULT=true`。
- 建库默认不加载 reranker，节约显存。

如果临时关闭查询 rerank：

```bash
RERANK_BY_DEFAULT=false python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --query-only \
  --query "你的问题"
```

## 6. 并发和超时

本地稳定默认值在 `configs/local-qwen3vl.env.example` 里。常用参数：

| 参数 | 作用 |
| --- | --- |
| `MAX_PARALLEL_INSERT` | 文档/文件 pipeline 并发 |
| `MAX_ASYNC` | 基础 LLM 调用并发；未设置 role-specific 时 extract/query/keyword 会回退到它 |
| `EXTRACT_MAX_ASYNC_LLM` | 实体/关系抽取的 LLM 并发，可手动加到 env |
| `VLM_MAX_ASYNC_LLM` | LightRAG 多模态 description/analyze 的 Qwen3-VL 并发 |
| `MAX_PARALLEL_ANALYZE` | 多模态 analyze worker 并发 |
| `EMBEDDING_FUNC_MAX_ASYNC` | embedding 调用并发 |
| `EMBEDDING_BATCH_NUM` | 单次 embedding batch |
| `MAX_ASYNC_RERANK` | reranker 并发 |
| `MAX_PARALLEL_PARSE_MINERU` | MinerU parse worker 并发 |

稳定推荐配置：

```env
MAX_PARALLEL_INSERT=2
MAX_ASYNC=8
EXTRACT_MAX_ASYNC_LLM=4
VLM_MAX_ASYNC_LLM=2
MAX_PARALLEL_ANALYZE=2
EMBEDDING_FUNC_MAX_ASYNC=4
EMBEDDING_BATCH_NUM=4
MAX_ASYNC_RERANK=1
MAX_PARALLEL_PARSE_MINERU=1
```

如果 Qwen3-VL 卡利用率不高且没有 timeout/OOM，可以逐步提高：

```env
VLM_MAX_ASYNC_LLM=3
MAX_PARALLEL_ANALYZE=3
```

再稳定后考虑：

```env
VLM_MAX_ASYNC_LLM=4
MAX_PARALLEL_ANALYZE=4
```

如果 entity/relation extraction 出现长时间卡住，先降低 extract 并发，不要只加 timeout：

```env
MAX_ASYNC=8
EXTRACT_MAX_ASYNC_LLM=4
```

`LLM_TIMEOUT` 是单次 LLM provider timeout。LightRAG worker timeout 约等于：

```text
LLM_TIMEOUT * 2
```

例如：

```env
LLM_TIMEOUT=1800
```

worker timeout 约为 3600 秒。若改成：

```env
LLM_TIMEOUT=3600
```

worker timeout 约为 7200 秒。1 小时仍无返回通常更像 Qwen3 vLLM 请求卡住或排队过久，应优先检查 vLLM 日志和降低抽取并发。

## 7. 数据隔离

默认不要改这些值：

```env
WORKSPACE=internal_lightrag
WORKING_DIR=/data/y50056788/Yaliang/internal_lightrag/rag_workspace
INPUT_DIR=/data/y50056788/Yaliang/internal_lightrag/inputs
REPORT_DIR=/data/y50056788/Yaliang/internal_lightrag/reports
QDRANT_COLLECTION_PREFIX=local_lightrag_bm25
```

Qdrant collection 名通常类似：

```text
local_lightrag_bm25_internal_lightrag_vdb_chunks_bge_m3_1024d
local_lightrag_bm25_internal_lightrag_vdb_entities_bge_m3_1024d
local_lightrag_bm25_internal_lightrag_vdb_relationships_bge_m3_1024d
```

Neo4j 使用 `internal_lightrag` workspace/label 隔离。不要把 `WORKSPACE`、`QDRANT_WORKSPACE` 或 `NEO4J_WORKSPACE` 改成开发版的 `internal`。

脚本不会写入开发版输出目录：

```text
/data/y50056788/Yaliang/internal/
```

开发版 LibreOffice PDF 默认只读复用：

```env
REUSE_DEV_LIBREOFFICE_PDFS=true
DEV_LIBREOFFICE_PDF_ROOT=/data/y50056788/Yaliang/internal/output/internal
```

复用命中后，脚本会把 PDF stage 到：

```text
/data/y50056788/Yaliang/internal_lightrag/inputs/internal_lightrag/
```

## 8. 常见问题

### tiktoken SSL 下载失败

脚本使用：

```env
LLM_TOKENIZER_MODEL_PATH=/data/y50056788/Yaliang/models/Qwen3-VL-30B-A3B-Instruct-FP8
```

并本地加载 HuggingFace tokenizer，避免下载 `o200k_base.tiktoken`。

### MinerU `/tasks` timeout

如果日志出现：

```text
MinerU local task polling timeout
```

说明 MinerU 接受了任务，但 LightRAG 客户端在轮询窗口内没有等到 completed。可增大：

```env
MINERU_POLL_INTERVAL_SECONDS=5
MINERU_MAX_POLLS=720
```

当前默认是 5 秒轮询一次，最多 720 次，约 1 小时。

### MinerU 显存超过 `--gpu-memory-utilization 0.1`

`--gpu-memory-utilization` 主要影响 MinerU 内部 vLLM EngineCore 的显存规划，不是整个 MinerU 进程组的硬限制。MinerU API 主进程、torch/transformers 模型、OCR/layout/table 模型和 CUDA cache 不一定受这个参数严格限制。

如果 MinerU 日志出现：

```text
Using transformers as the inference engine for VLM.
```

说明它没有走 vLLM VLM engine。显存紧张或 transformers 崩溃时，优先尝试：

```env
MINERU_LOCAL_IMAGE_ANALYSIS=false
```

### parser routing 写法

官方 parser routing 使用无点后缀，不是 shell glob：

```env
LIGHTRAG_PARSER=pdf:mineru-iteP,docx:mineru-iteP,txt:legacy-F,md:legacy-F
```

不要写：

```env
*.pdf:mineru-iteP
```

旧 env 中的 `*.pdf`、`.pdf` 写法会被脚本尽量规范化，但建议直接使用官方格式。

### `tif/tiff` 不支持

当前官方 MinerU routing 不支持 `tif/tiff`。本地脚本默认不会扫描这两类文件；旧 env 里残留的 `tif/tiff:mineru-...` 规则会被脚本自动丢弃。

### embedding meta tensor 报错

本地 embedding 是进程内 `SentenceTransformer`，不是远端 embedding server。脚本已对首次模型加载加线程锁，避免多个 worker 同时加载 CUDA 模型触发 meta tensor 问题；模型加载完成后仍按 `EMBEDDING_FUNC_MAX_ASYNC` 并发处理 embedding。

### References 编号不连续

查询回答里的 `[2]`、`[5]`、`[8]` 是本次上下文里的原始 `reference_id`，不保证连续。缺失的编号通常表示对应候选 source 没有被最终答案引用。

## 9. GitLab 上传

Linux 上从 GitHub pull 到最新后，添加 GitLab remote：

```bash
cd /data/y50056788/Yaliang/LightRAG-0528
git pull origin main
git remote add gitlab <你的GitLab仓库URL>
git push gitlab main
```

如果 GitLab 空仓库已经存在，直接 push `main` 即可。

如果需要保留 GitHub remote，不要覆盖 `origin`，使用新的 remote 名称 `gitlab`。

如果 `gitlab` remote 已经存在：

```bash
git remote set-url gitlab <你的GitLab仓库URL>
git push gitlab main
```

## 10. 验证命令

代码编译检查：

```bash
python -m py_compile scripts/build_internal_lightrag.py local_lightrag/qdrant_hybrid_bm25.py
```

adapter 测试：

```bash
python -m pytest tests/test_build_internal_lightrag_adapter.py -q
```

dry-run 检查：

```bash
python scripts/build_internal_lightrag.py \
  --env-file configs/local-qwen3vl.env \
  --dry-run \
  --max-files 10
```

更多细节见 [docs/local-linux-build.md](docs/local-linux-build.md)。
