from __future__ import annotations

from unittest.mock import MagicMock

import fakeredis
import pytest
import redis as redis_lib

from app.config import Settings
from app.foundation.llm.runtime_config import (
    HASH_KEY,
    MODEL_KEYS,
    RuntimeConfigUnavailable,
    RuntimeModelConfig,
)


def make_settings(**overrides) -> Settings:
    defaults = {
        "OPENAI_API_KEY": "sk-test",
        "PRIMARY_MODEL": "gpt-4o-mini",
        "LLM_PROVIDER": "openai",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})


@pytest.fixture
def store() -> RuntimeModelConfig:
    return RuntimeModelConfig(
        fakeredis.FakeRedis(decode_responses=True), make_settings()
    )


def test_effective_returns_settings_default_when_no_override(store) -> None:
    assert store.effective("PRIMARY_MODEL") == "gpt-4o-mini"
    assert store.is_overridden("PRIMARY_MODEL") is False


def test_set_and_effective_round_trip(store) -> None:
    store.set("PRIMARY_MODEL", "gpt-4o")
    assert store.effective("PRIMARY_MODEL") == "gpt-4o"
    assert store.is_overridden("PRIMARY_MODEL") is True


def test_set_none_resets_to_default(store) -> None:
    store.set("PRIMARY_MODEL", "gpt-4o")
    store.set("PRIMARY_MODEL", None)
    assert store.effective("PRIMARY_MODEL") == "gpt-4o-mini"


def test_unknown_key_raises(store) -> None:
    with pytest.raises(ValueError, match="Unknown model key"):
        store.effective("EMBEDDING_MODEL")


def test_snapshot_shape_covers_every_key(store) -> None:
    store.set("CRITIC_MODEL", "gpt-4o")
    snapshot = store.snapshot()
    assert set(snapshot) == set(MODEL_KEYS)
    assert snapshot["CRITIC_MODEL"]["overridden"] is True


def test_reads_degrade_to_defaults_when_redis_down() -> None:
    broken = MagicMock()
    broken.hget.side_effect = redis_lib.RedisError("down")
    broken.hgetall.side_effect = redis_lib.RedisError("down")
    store = RuntimeModelConfig(broken, make_settings())
    assert store.effective("PRIMARY_MODEL") == "gpt-4o-mini"


def test_writes_raise_when_redis_down() -> None:
    broken = MagicMock()
    broken.hset.side_effect = redis_lib.RedisError("down")
    store = RuntimeModelConfig(broken, make_settings())
    with pytest.raises(RuntimeConfigUnavailable):
        store.set("PRIMARY_MODEL", "gpt-4o")


def test_overrides_share_the_hash_key(store) -> None:
    store.set("PRIMARY_MODEL", "gpt-4o")
    assert store._redis.hgetall(HASH_KEY) == {"PRIMARY_MODEL": "gpt-4o"}
