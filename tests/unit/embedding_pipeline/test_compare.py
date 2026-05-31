import pytest

from scripts.compare import cosine_similarity


def test_cosine_similarity_for_identical_direction():
    assert cosine_similarity([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_rejects_zero_vectors():
    with pytest.raises(ValueError, match="zero vectors"):
        cosine_similarity([0.0, 0.0], [1.0, 1.0])
