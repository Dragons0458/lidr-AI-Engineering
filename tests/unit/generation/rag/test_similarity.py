import pytest

from app.generation.rag.analysis.similarity import cosine_similarity, percentile


def test_cosine_similarity_identical_direction():
    assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_norm_returns_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_percentile_empty():
    assert percentile([], 50) == 0.0


def test_percentile_single_value():
    assert percentile([42], 50) == 42.0


def test_percentile_p50_and_p95():
    values = [10, 20, 30, 40, 50]
    assert percentile(values, 50) == pytest.approx(30.0)
    assert percentile(values, 95) == pytest.approx(48.0)
