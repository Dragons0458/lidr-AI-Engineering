import json
from types import SimpleNamespace

import pytest

from app.cache.semantic import EstimationSemanticCache
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)


def _request(
    description: str = "Portal web con login y reportes.",
) -> EstimationRequest:
    return EstimationRequest(
        description=description,
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )


@pytest.fixture
def semantic_cache(monkeypatch):
    class FakeVectorizer:
        def embed(self, text: str) -> list[float]:
            return [0.1, 0.2, 0.3] if "similar" in text else [0.9, 0.8, 0.7]

    fake_index = SimpleNamespace(
        create_calls=0,
        query_results=None,
        loaded=None,
    )

    def create(overwrite=False):
        fake_index.create_calls += 1

    def query(_query):
        return fake_index.query_results or []

    def load(records, ttl=None):
        fake_index.loaded = {"records": records, "ttl": ttl}

    fake_index.create = create
    fake_index.query = query
    fake_index.load = load
    fake_index.set_client = lambda *_args, **_kwargs: None

    monkeypatch.setattr(
        "app.cache.semantic.SearchIndex.from_dict",
        lambda _schema: fake_index,
    )

    cache = EstimationSemanticCache(
        redis_client=object(),
        vectorizer=FakeVectorizer(),
        threshold=0.92,
        ttl=3600,
        log_only=False,
    )
    cache._fake_index = fake_index
    return cache


def test_bucket_for_four_fields(semantic_cache) -> None:
    bucket = semantic_cache.bucket_for(_request(), "v1")
    assert bucket == "v1:web_saas:medium:line_items"


def test_lookup_miss_below_threshold(semantic_cache) -> None:
    semantic_cache._fake_index.query_results = [
        {"result_json": "{}", "vector_distance": 0.2}
    ]
    assert semantic_cache.lookup(_request("different text"), "v1") is None


def test_lookup_hit_above_threshold(semantic_cache) -> None:
    payload = {"estimation": "cached", "model": "gpt-4o-mini"}
    semantic_cache._fake_index.query_results = [
        {
            "result_json": json.dumps(payload),
            "vector_distance": 0.05,
        }
    ]

    hit = semantic_cache.lookup(_request("similar description"), "v1")
    assert hit == payload


def test_log_only_does_not_return_hit(semantic_cache) -> None:
    semantic_cache.log_only = True
    semantic_cache._fake_index.query_results = [
        {
            "result_json": json.dumps({"estimation": "cached"}),
            "vector_distance": 0.01,
        }
    ]

    assert semantic_cache.lookup(_request("similar description"), "v1") is None


def test_store_calls_index_load_with_ttl(semantic_cache) -> None:
    payload = {"estimation": "stored", "model": "gpt-4o-mini"}
    semantic_cache.store(_request(), payload, "v2")

    assert semantic_cache._fake_index.loaded is not None
    assert semantic_cache._fake_index.loaded["ttl"] == 3600
    record = semantic_cache._fake_index.loaded["records"][0]
    assert record["bucket"] == "v2:web_saas:medium:line_items"
    assert json.loads(record["result_json"]) == payload
    assert isinstance(record["embedding"], bytes)
