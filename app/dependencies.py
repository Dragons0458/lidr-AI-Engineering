from functools import lru_cache

from app.config import get_settings
from app.services.cache import EstimationCache
from app.services.llm_wrapper import LLMWrapper


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
