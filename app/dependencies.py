from functools import lru_cache

import redis
import structlog
from redisvl.utils.vectorize import OpenAITextVectorizer

from app.cache.semantic import EstimationSemanticCache
from app.config import get_settings
from app.ingestion.catalog import DataCatalog, load_catalog
from app.ingestion.loaders.filesystem import FileSystemLoader
from app.ingestion.parsers.registry import ParserRegistry, default_registry
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper

log = structlog.get_logger()


@lru_cache
def get_cache() -> EstimationCache:
    settings = get_settings()
    return EstimationCache.from_url(settings.REDIS_URL, ttl=settings.CACHE_TTL)


@lru_cache
def get_llm_wrapper() -> LLMWrapper:
    settings = get_settings()
    primary = settings.PRIMARY_MODEL
    return LLMWrapper(
        primary_model=primary,
        fallback_model=settings.FALLBACK_MODEL,
        timeout=settings.LLM_TIMEOUT,
        num_retries=settings.LLM_RETRIES,
        cache=get_cache(),
        cache_enabled=settings.CACHE_ENABLED,
    )


@lru_cache
def get_semantic_cache() -> EstimationSemanticCache | None:
    """Return semantic cache or None when disabled or Redis/embedding setup fails."""
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
    """Build a ConsistentPseudonymizer backed by Postgres (Session 6).

    Not a singleton — the mapping store wraps a Session, so callers pass their own.
    """
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
