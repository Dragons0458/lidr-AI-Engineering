"""Semantic similarity cache for estimations using RedisVL vector search."""

from __future__ import annotations

import json
from typing import Any, Protocol

import numpy as np
import structlog
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from app.schemas.estimation import EstimationRequest

log = structlog.get_logger()

_INDEX_SCHEMA: dict[str, Any] = {
    "index": {
        "name": "estimations",
        "prefix": "estimation_semantic",
        "storage_type": "hash",
    },
    "fields": [
        {"name": "bucket", "type": "tag"},
        {"name": "result_json", "type": "text"},
        {
            "name": "embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "flat",
                "dims": 1536,
                "distance_metric": "cosine",
            },
        },
    ],
}


def _to_bytes(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


class _TextVectorizer(Protocol):
    def embed(self, text: str) -> list[float]: ...


class EstimationSemanticCache:
    """RedisVL-backed semantic cache keyed by prompt bucket and description embedding."""

    def __init__(
        self,
        *,
        redis_client: Any,
        vectorizer: _TextVectorizer,
        threshold: float = 0.92,
        ttl: int = 86400,
        log_only: bool = False,
        index_name: str = "estimations",
    ) -> None:
        self.threshold = threshold
        self.ttl = ttl
        self.log_only = log_only
        self.vectorizer = vectorizer
        schema = {
            **_INDEX_SCHEMA,
            "index": {**_INDEX_SCHEMA["index"], "name": index_name},
        }
        self.index = SearchIndex.from_dict(schema)
        self.index.set_client(redis_client)
        try:
            self.index.create(overwrite=False)
        except Exception as exc:
            message = str(exc).lower()
            if "already" not in message and "exists" not in message:
                raise

    def bucket_for(self, request: EstimationRequest, prompt_version: str) -> str:
        return (
            f"{prompt_version}:{request.project_type.value}:"
            f"{request.detail_level.value}:{request.output_format.value}"
        )

    def lookup(
        self, request: EstimationRequest, prompt_version: str
    ) -> dict[str, Any] | None:
        bucket = self.bucket_for(request, prompt_version)
        embedding = self.vectorizer.embed(request.description)
        query = VectorQuery(
            vector=embedding,
            vector_field_name="embedding",
            return_fields=["result_json", "vector_distance"],
            filter_expression=Tag("bucket") == bucket,
            num_results=1,
            return_score=True,
        )
        try:
            results = self.index.query(query)
        except Exception as exc:
            log.warning("semantic_cache_lookup_failed", error=str(exc))
            return None

        if not results:
            log.info("semantic_cache_miss", bucket=bucket, reason="no_results")
            return None

        doc = results[0]
        distance = float(doc.get("vector_distance", 1.0))
        similarity = 1.0 - distance
        log.info(
            "semantic_cache_candidate",
            bucket=bucket,
            similarity=round(similarity, 4),
            threshold=self.threshold,
        )

        if similarity < self.threshold:
            log.info(
                "semantic_cache_miss",
                bucket=bucket,
                similarity=round(similarity, 4),
                threshold=self.threshold,
                hint=(
                    "below_threshold"
                    if similarity >= self.threshold - 0.05
                    else "low_similarity"
                ),
            )
            return None

        if self.log_only:
            log.info(
                "semantic_cache_log_only_hit",
                bucket=bucket,
                similarity=similarity,
            )
            return None

        raw = doc.get("result_json")
        if not raw:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning("semantic_cache_payload_invalid", error=str(exc))
            return None

    def store(
        self,
        request: EstimationRequest,
        payload: dict[str, Any],
        prompt_version: str,
    ) -> None:
        bucket = self.bucket_for(request, prompt_version)
        embedding = self.vectorizer.embed(request.description)
        record = {
            "bucket": bucket,
            "result_json": json.dumps(payload),
            "embedding": _to_bytes(embedding),
        }
        try:
            self.index.load([record], ttl=self.ttl)
            log.info("semantic_cache_stored", bucket=bucket, ttl=self.ttl)
        except Exception as exc:
            log.warning("semantic_cache_store_failed", error=str(exc))
