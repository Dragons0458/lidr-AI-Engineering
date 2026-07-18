"""FastAPI dependency factories for shared singletons."""

from __future__ import annotations

from functools import lru_cache

import anthropic
import redis
import structlog
from openai import AsyncOpenAI, OpenAI
from redisvl.utils.vectorize import OpenAITextVectorizer

from app.config import get_settings
from app.foundation.llm.runtime_config import RuntimeModelConfig, RuntimeRetrievalConfig
from app.foundation.llm.wrapper import LLMWrapper
from app.generation.cag.exact import EstimationCache
from app.generation.cag.semantic import EstimationSemanticCache
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.chunking.structural import JSONStructuralChunker
from app.generation.rag.chunking.strategies import (
    ContextualRetrievalChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    PropositionalChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceWindowChunker,
)
from app.foundation.persistence.async_database import get_async_session_factory
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.ingest_service import RagIngestService
from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.store.repository import ChunkStore
from app.ingestion.catalog import DataCatalog, load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import ParserRegistry, default_registry

log = structlog.get_logger()


@lru_cache
def get_graph_activity():
    from app.generation.agentic.graph.activity import GraphActivityLog

    return GraphActivityLog.from_settings(get_settings())


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_runtime_config() -> RuntimeModelConfig:
    """Redis-backed override store for the LLM model knobs (Settings UI)."""
    settings = get_settings()
    return RuntimeModelConfig.from_url(settings.REDIS_URL, settings)


@lru_cache
def get_runtime_retrieval_config() -> RuntimeRetrievalConfig:
    """Redis-backed override store for Session 10 retrieval toggles."""
    settings = get_settings()
    return RuntimeRetrievalConfig.from_url(settings.REDIS_URL, settings)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    return LLMWrapper(
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
        cache_enabled=settings.CACHE_ENABLED,
        runtime_config=get_runtime_config(),
    )


@lru_cache
def get_openai_client() -> OpenAI | None:
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return OpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_async_openai_client() -> AsyncOpenAI | None:
    """Lazy async OpenAI client for the Session 12 agentic loop.

    The agent (``app/generation/agentic/agent_loop.py``) drives the raw Responses
    API (``client.responses.create``) with ``await``, alongside the async
    ``retrieve()`` its ``search_budgets`` tool wraps — so it needs an async client,
    not the sync one used by moderation/embeddings. Returns ``None`` when no
    OpenAI key is configured (the agent needs OpenAI specifically for the
    Responses API).
    """
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        return None
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


@lru_cache
def get_chunker() -> JSONStructuralChunker:
    return JSONStructuralChunker()


@lru_cache
def get_embedder() -> OpenAIEmbedder | None:
    settings = get_settings()
    client = get_openai_client()
    if client is None:
        log.warning("embedder_disabled", reason="no_openai_key")
        return None
    return OpenAIEmbedder(client=client, model=settings.EMBEDDING_MODEL)


@lru_cache
def get_chunk_store() -> ChunkStore:
    return ChunkStore()


@lru_cache
def get_rag_ingest_service() -> RagIngestService | None:
    embedder = get_embedder()
    if embedder is None:
        return None
    return RagIngestService(
        chunker=get_chunker(),
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


@lru_cache
def get_corpus_index_service() -> CorpusIndexService | None:
    ingest = get_rag_ingest_service()
    if ingest is None:
        return None
    return CorpusIndexService(ingest=ingest)


@lru_cache
def get_reranker():
    from app.generation.rag.retrieval.reranker import CrossEncoderReranker

    settings = get_settings()
    return CrossEncoderReranker(settings.RERANKER_MODEL)


@lru_cache
def get_semantic_retriever() -> SemanticRetriever | None:
    embedder = get_embedder()
    if embedder is None:
        return None
    return SemanticRetriever(
        embedder=embedder,
        session_factory=get_async_session_factory(),
        store=get_chunk_store(),
    )


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic | None:
    settings = get_settings()
    if not settings.ANTHROPIC_API_KEY:
        return None
    return anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


@lru_cache
def get_fixed_size_chunker() -> FixedSizeChunker:
    return FixedSizeChunker()


@lru_cache
def get_recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker()


@lru_cache
def get_sentence_window_chunker() -> SentenceWindowChunker:
    return SentenceWindowChunker()


@lru_cache
def get_hierarchical_chunker() -> HierarchicalChunker:
    return HierarchicalChunker()


@lru_cache
def get_semantic_chunker() -> SemanticChunker:
    settings = get_settings()
    return SemanticChunker(
        api_key=settings.OPENAI_API_KEY, model=settings.EMBEDDING_MODEL
    )


def get_propositional_chunker() -> PropositionalChunker:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("PropositionalChunker requires OPENAI_API_KEY.")
    model = get_runtime_config().effective("PROPOSITIONAL_CHUNKER_MODEL")
    return PropositionalChunker(client=client, model=model)


def get_contextual_retrieval_chunker() -> ContextualRetrievalChunker:
    client = get_anthropic_client()
    if client is None:
        raise RuntimeError("ContextualRetrievalChunker requires ANTHROPIC_API_KEY.")
    model = get_runtime_config().effective("CONTEXTUAL_CHUNKER_MODEL")
    return ContextualRetrievalChunker(client=client, model=model)


CHUNKER_FACTORIES = {
    "structural": get_chunker,
    "fixed_size": get_fixed_size_chunker,
    "recursive": get_recursive_chunker,
    "sentence_window": get_sentence_window_chunker,
    "semantic": get_semantic_chunker,
    "propositional": get_propositional_chunker,
    "contextual_retrieval": get_contextual_retrieval_chunker,
    "hierarchical": get_hierarchical_chunker,
}
ALL_STRATEGIES = list(CHUNKER_FACTORIES)


def build_chunkers(names: list[str]) -> list[Chunker]:
    chunkers: list[Chunker] = []
    for name in names:
        factory = CHUNKER_FACTORIES.get(name)
        if factory is None:
            raise KeyError(name)
        chunkers.append(factory())
    return chunkers


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    settings = get_settings()
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None
    if not settings.OPENAI_API_KEY:
        log.info("semantic_cache_disabled", reason="missing_openai_api_key")
        return None

    try:
        vectorizer = OpenAITextVectorizer(
            model=settings.EMBEDDING_MODEL,
            api_config={"api_key": settings.OPENAI_API_KEY},
        )
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        return EstimationSemanticCache(
            redis_client=redis_client,
            vectorizer=vectorizer,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
            ttl=settings.SEMANTIC_CACHE_TTL,
            log_only=settings.SEMANTIC_CACHE_LOG_ONLY,
        )
    except Exception as exc:
        log.warning("semantic_cache_init_failed", error=str(exc))
        return None


def build_pseudonymizer(session):
    """Build a ConsistentPseudonymizer backed by Postgres (Session 6)."""
    from app.ingestion.pii import (
        ConsistentPseudonymizer,
        PostgresMappingStore,
        build_analyzer,
    )

    settings = get_settings()
    return ConsistentPseudonymizer(
        analyzer=build_analyzer(),
        mapping_store=PostgresMappingStore(session),
        salt=settings.PSEUDONYM_HASH_SALT,
        faker_locale=settings.PSEUDONYM_FAKER_LOCALE,
        language="es",
    )


@lru_cache
def get_catalog() -> DataCatalog:
    settings = get_settings()
    return load_catalog(settings.CATALOG_PATH)


@lru_cache
def get_filesystem_loader() -> FileSystemLoader:
    settings = get_settings()
    return FileSystemLoader(data_root=settings.INGESTION_DATA_ROOT)


@lru_cache
def get_parser_registry() -> ParserRegistry:
    return default_registry()


@lru_cache
def get_idempotency_store():
    from app.generation.rag.idempotency import IdempotencyStore

    return IdempotencyStore.from_settings(get_settings())


@lru_cache
def get_token_encoder():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")
