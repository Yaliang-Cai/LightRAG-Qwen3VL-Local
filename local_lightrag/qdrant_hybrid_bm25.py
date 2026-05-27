from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from lightrag.kg.qdrant_impl import (
    CREATED_AT_FIELD,
    DEFAULT_QDRANT_UPSERT_MAX_POINTS_PER_BATCH,
    ID_FIELD,
    WORKSPACE_ID_FIELD,
    QdrantVectorDBStorage,
    compute_mdhash_id_for_qdrant,
    config,
    workspace_filter_condition,
)
from lightrag.kg.shared_storage import get_data_init_lock
from lightrag.utils import logger

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower() or "default"


def _sparse_vector_from_embedding(embedding: Any) -> models.SparseVector:
    indices = getattr(embedding, "indices", None)
    values = getattr(embedding, "values", None)
    if indices is None or values is None:
        raise TypeError("Sparse embedding result must expose indices and values")
    return models.SparseVector(
        indices=[int(index) for index in indices],
        values=[float(value) for value in values],
    )


@dataclass
class QdrantHybridBM25VectorDBStorage(QdrantVectorDBStorage):
    """Qdrant storage with dense vectors plus BM25 sparse vectors.

    This adapter is intentionally local to this repository. It keeps official
    LightRAG source untouched while adding Qdrant hybrid indexing for the
    Qwen3-VL local build profile.
    """

    def __post_init__(self):
        super().__post_init__()
        self.enable_sparse_bm25 = _truthy(
            os.getenv("QDRANT_ENABLE_SPARSE_BM25"), default=True
        )
        self.retrieval_mode = (
            os.getenv("QDRANT_RETRIEVAL_MODE", "hybrid").strip().lower()
        )
        if self.retrieval_mode not in {"dense", "bm25", "hybrid"}:
            logger.warning(
                "Unsupported QDRANT_RETRIEVAL_MODE=%s; falling back to hybrid",
                self.retrieval_mode,
            )
            self.retrieval_mode = "hybrid"

        self.sparse_model_name = os.getenv("QDRANT_SPARSE_BM25_MODEL", "Qdrant/bm25")
        self._sparse_model = None
        prefix = _safe_name(os.getenv("QDRANT_COLLECTION_PREFIX", "local_lightrag_bm25"))
        workspace = _safe_name(self.effective_workspace)
        suffix = self.model_suffix or "local_embedding"
        self.final_namespace = f"{prefix}_{workspace}_vdb_{self.namespace}_{suffix}"
        logger.info(
            "Qdrant hybrid BM25 collection: %s mode=%s sparse=%s model=%s",
            self.final_namespace,
            self.retrieval_mode,
            self.enable_sparse_bm25,
            self.sparse_model_name,
        )

    def _get_sparse_model(self):
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding

            cache_dir = os.getenv("QDRANT_FASTEMBED_CACHE_DIR") or None
            threads_env = os.getenv("QDRANT_BM25_THREADS", "").strip()
            threads = int(threads_env) if threads_env else None
            logger.info(
                "Loading Qdrant sparse BM25 model: %s cache_dir=%s threads=%s",
                self.sparse_model_name,
                cache_dir or "<fastembed default>",
                threads,
            )
            self._sparse_model = SparseTextEmbedding(
                self.sparse_model_name,
                cache_dir=cache_dir,
                threads=threads,
                lazy_load=True,
            )
        return self._sparse_model

    async def _encode_sparse(self, texts: list[str]) -> list[models.SparseVector]:
        if not texts:
            return []

        def encode() -> list[models.SparseVector]:
            model = self._get_sparse_model()
            return [_sparse_vector_from_embedding(item) for item in model.embed(texts)]

        return await asyncio.to_thread(encode)

    async def initialize(self):
        async with get_data_init_lock():
            if self._initialized:
                return
            try:
                if self._client is None:
                    self._client = QdrantClient(
                        url=os.environ.get(
                            "QDRANT_URL", config.get("qdrant", "uri", fallback=None)
                        ),
                        api_key=os.environ.get(
                            "QDRANT_API_KEY",
                            config.get("qdrant", "apikey", fallback=None),
                        ),
                    )

                if not self._client.collection_exists(self.final_namespace):
                    sparse_config = None
                    if self.enable_sparse_bm25:
                        sparse_config = {
                            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                                index=models.SparseIndexParams(on_disk=False),
                                modifier=models.Modifier.IDF,
                            )
                        }
                    self._client.create_collection(
                        self.final_namespace,
                        vectors_config={
                            DENSE_VECTOR_NAME: models.VectorParams(
                                size=self.embedding_func.embedding_dim,
                                distance=models.Distance.COSINE,
                            )
                        },
                        sparse_vectors_config=sparse_config,
                        hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
                    )
                    logger.info(
                        "Qdrant hybrid collection '%s' created", self.final_namespace
                    )

                self._client.create_payload_index(
                    collection_name=self.final_namespace,
                    field_name=WORKSPACE_ID_FIELD,
                    field_schema=models.KeywordIndexParams(
                        type=models.KeywordIndexType.KEYWORD,
                        is_tenant=True,
                    ),
                )
                self._initialized = True
            except Exception as exc:
                logger.error(
                    "[%s] Failed to initialize Qdrant hybrid collection '%s': %s",
                    self.workspace,
                    self.final_namespace,
                    exc,
                )
                raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.debug(
            "[%s] Hybrid Qdrant inserting %d to %s",
            self.workspace,
            len(data),
            self.namespace,
        )
        if not data:
            return

        current_time = int(time.time())
        list_data = [
            {
                ID_FIELD: key,
                WORKSPACE_ID_FIELD: self.effective_workspace,
                CREATED_AT_FIELD: current_time,
                **{
                    field: value
                    for field, value in item.items()
                    if field in self.meta_fields
                },
            }
            for key, item in data.items()
        ]
        contents = [item["content"] for item in data.values()]
        batches = [
            contents[index : index + self._max_batch_size]
            for index in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch, context="document") for batch in batches]
        )
        dense_embeddings = np.concatenate(embeddings_list)
        sparse_embeddings = (
            await self._encode_sparse(contents)
            if self.enable_sparse_bm25
            else [None] * len(contents)
        )

        points: list[models.PointStruct] = []
        for index, payload in enumerate(list_data):
            vector: dict[str, Any] = {
                DENSE_VECTOR_NAME: np.asarray(
                    dense_embeddings[index], dtype=np.float32
                ).tolist()
            }
            if self.enable_sparse_bm25:
                vector[SPARSE_VECTOR_NAME] = sparse_embeddings[index]
            points.append(
                models.PointStruct(
                    id=compute_mdhash_id_for_qdrant(
                        payload[ID_FIELD], prefix=self.effective_workspace
                    ),
                    vector=vector,
                    payload=payload,
                )
            )

        max_points = int(
            os.getenv(
                "QDRANT_UPSERT_MAX_POINTS_PER_BATCH",
                str(DEFAULT_QDRANT_UPSERT_MAX_POINTS_PER_BATCH),
            )
        )
        max_points = max(1, max_points)
        for start in range(0, len(points), max_points):
            self._client.upsert(
                collection_name=self.final_namespace,
                points=points[start : start + max_points],
                wait=True,
            )

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        if query_embedding is not None:
            dense_embedding = query_embedding
        else:
            embedding_result = await self.embedding_func(
                [query], context="query", _priority=5
            )
            dense_embedding = np.asarray(embedding_result[0], dtype=np.float32).tolist()

        query_filter = models.Filter(
            must=[workspace_filter_condition(self.effective_workspace)]
        )
        mode = self.retrieval_mode
        if not self.enable_sparse_bm25 and mode in {"bm25", "hybrid"}:
            mode = "dense"

        if mode == "bm25":
            sparse_query = (await self._encode_sparse([query]))[0]
            response = self._client.query_points(
                collection_name=self.final_namespace,
                query=sparse_query,
                using=SPARSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
            )
        elif mode == "hybrid":
            sparse_query = (await self._encode_sparse([query]))[0]
            prefetch_limit = max(
                top_k,
                top_k * int(os.getenv("QDRANT_HYBRID_PREFETCH_MULTIPLIER", "4")),
            )
            response = self._client.query_points(
                collection_name=self.final_namespace,
                prefetch=[
                    models.Prefetch(
                        query=dense_embedding,
                        using=DENSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=prefetch_limit,
                        score_threshold=self.cosine_better_than_threshold,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=SPARSE_VECTOR_NAME,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self._client.query_points(
                collection_name=self.final_namespace,
                query=dense_embedding,
                using=DENSE_VECTOR_NAME,
                limit=top_k,
                with_payload=True,
                score_threshold=self.cosine_better_than_threshold,
                query_filter=query_filter,
            )

        return [
            {
                **point.payload,
                "distance": point.score,
                CREATED_AT_FIELD: point.payload.get(CREATED_AT_FIELD),
            }
            for point in response.points
        ]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        try:
            qdrant_ids = [
                compute_mdhash_id_for_qdrant(item_id, prefix=self.effective_workspace)
                for item_id in ids
            ]
            results = self._client.retrieve(
                collection_name=self.final_namespace,
                ids=qdrant_ids,
                with_vectors=[DENSE_VECTOR_NAME],
                with_payload=True,
            )
            vectors: dict[str, list[float]] = {}
            for point in results:
                payload = point.payload or {}
                original_id = payload.get(ID_FIELD)
                if not original_id:
                    continue
                vector_data = point.vector
                if isinstance(vector_data, dict):
                    vector_data = vector_data.get(DENSE_VECTOR_NAME)
                if isinstance(vector_data, np.ndarray):
                    vector_data = vector_data.tolist()
                if vector_data is not None:
                    vectors[str(original_id)] = list(vector_data)
            return vectors
        except Exception as exc:
            logger.error(
                "[%s] Error retrieving dense vectors by IDs from %s: %s",
                self.workspace,
                self.namespace,
                exc,
            )
            return {}
