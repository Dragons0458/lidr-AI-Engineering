"""Unit tests for RuntimeRetrievalConfig (Session 10)."""

from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import pytest
import redis as redis_lib

from app.config import Settings
from app.foundation.llm.runtime_config import (
    RETRIEVAL_HASH_KEY,
    RETRIEVAL_KEYS,
    RuntimeConfigUnavailable,
    RuntimeRetrievalConfig,
)


def make_settings(**overrides) -> Settings:
    return Settings(OPENAI_API_KEY="sk-test", _env_file=None, **overrides)


@pytest.fixture
def store() -> RuntimeRetrievalConfig:
    return RuntimeRetrievalConfig(
        fakeredis.FakeRedis(decode_responses=True), make_settings()
    )


def test_effective_defaults_without_override(store) -> None:
    assert store.effective_search_mode() == "vector"
    assert store.effective_rerank() is False
    assert store.effective_routing() is True


def test_search_mode_override(store) -> None:
    store.set_search_mode("hybrid")
    assert store.effective_search_mode() == "hybrid"


def test_rerank_override(store) -> None:
    store.set_rerank(True)
    assert store.effective_rerank() is True
    store.set_rerank(None)
    assert store.effective_rerank() is False


def test_snapshot_shape_covers_every_key(store) -> None:
    snapshot = store.snapshot()
    assert set(snapshot) == set(RETRIEVAL_KEYS)
    assert snapshot["RETRIEVAL_SEARCH_MODE"]["overridden"] is False


def test_s11_toggle_defaults(store) -> None:
    assert store.effective_hallucination_gate() is True
    assert store.effective_augmentation() is True
    assert store.effective_synthesis() is True


def test_s11_set_bool_overrides(store) -> None:
    from app.foundation.llm.runtime_config import (
        AUGMENTATION_KEY,
        HALLUCINATION_GATE_KEY,
        SYNTHESIS_KEY,
    )

    store.set_bool(HALLUCINATION_GATE_KEY, False)
    store.set_bool(AUGMENTATION_KEY, False)
    store.set_bool(SYNTHESIS_KEY, False)
    assert store.effective_hallucination_gate() is False
    assert store.effective_augmentation() is False
    assert store.effective_synthesis() is False


def test_reads_degrade_when_redis_down() -> None:
    broken = MagicMock()
    broken.hget.side_effect = redis_lib.RedisError("down")
    broken.hgetall.side_effect = redis_lib.RedisError("down")
    store = RuntimeRetrievalConfig(broken, make_settings())
    assert store.effective_search_mode() == "vector"


def test_writes_raise_when_redis_down() -> None:
    broken = MagicMock()
    broken.hset.side_effect = redis_lib.RedisError("down")
    store = RuntimeRetrievalConfig(broken, make_settings())
    with pytest.raises(RuntimeConfigUnavailable):
        store.set_search_mode("hybrid")


def test_invalid_search_mode_raises(store) -> None:
    with pytest.raises(ValueError, match="Invalid search mode"):
        store.set_search_mode("invalid")


def test_overrides_share_retrieval_hash_key(store) -> None:
    store.set_search_mode("hybrid")
    assert store._redis.hgetall(RETRIEVAL_HASH_KEY) == {
        "RETRIEVAL_SEARCH_MODE": "hybrid"
    }
