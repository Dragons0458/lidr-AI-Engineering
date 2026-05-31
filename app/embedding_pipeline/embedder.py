from __future__ import annotations

import time

from openai import OpenAI, RateLimitError
import structlog

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

PRICE_PER_MILLION_TOKENS_USD = 0.02  # text-embedding-3-small, may change.
BATCH_SIZE = 100
MAX_RATE_LIMIT_RETRIES = 3

log = structlog.get_logger()


def estimate_embedding_cost_usd(total_tokens: int) -> float:
    return total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD


class OpenAIEmbedder:
    """OpenAI SDK embedder used for explicit batching and retry mechanics.

    The semantic cache uses redisvl.OpenAITextVectorizer elsewhere. This class is
    deliberately direct to make the Session 7 embedding pipeline easy to inspect.
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_one(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        embedded_chunks: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            vectors = self._embed_batch(batch)
            for chunk, vector in zip(batch, vectors, strict=True):
                embedded_chunks.append(
                    EmbeddedChunk(
                        **chunk.model_dump(),
                        embedding=vector,
                    )
                )
        return embedded_chunks

    def _embed_batch(self, batch: list[Chunk]) -> list[list[float]]:
        total_tokens = sum(chunk.token_count for chunk in batch)
        started = time.perf_counter()
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[chunk.text for chunk in batch],
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                log.info(
                    "embeddings_batch_done",
                    n_chunks=len(batch),
                    n_tokens=total_tokens,
                    latency_ms=latency_ms,
                )
                data = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in data]
            except RateLimitError:
                if attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("unreachable embedding retry state")
