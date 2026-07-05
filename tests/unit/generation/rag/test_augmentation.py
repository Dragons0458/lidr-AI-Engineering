"""Unit tests for deterministic chunk augmentation (Session 11)."""

from __future__ import annotations

from app.generation.rag.quality.augmentation import (
    augment_chunks,
    compress_chunk,
    extract_key_points,
    reorder_edge_loaded,
)
from app.generation.rag.schemas import RetrievedChunk


def _chunk(cid: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        content=content,
        chunk_type="budget_component",
        distance=0.1,
    )


def test_extract_key_points_keeps_figures_and_drops_filler():
    chunk = _chunk(
        1,
        "Narrative fluff about the project.\n"
        "Auth module :: OAuth integration\n"
        "Estimated effort: 12 engineer-days",
    )
    result = extract_key_points(chunk)
    assert "12" in result
    assert "::" in result
    assert "Narrative fluff" not in result


def test_extract_key_points_never_empty():
    chunk = _chunk(2, "   \n  ")
    assert extract_key_points(chunk)


def test_compress_chunk_preserves_id():
    chunk = _chunk(7, "line with 40 hours")
    compressed = compress_chunk(chunk)
    assert compressed.id == 7
    assert compressed.content != chunk.content or "40" in compressed.content


def test_reorder_edge_loaded_places_first_at_head_second_at_tail():
    chunks = [_chunk(i, f"chunk-{i}") for i in range(1, 5)]
    ordered = reorder_edge_loaded(chunks)
    assert ordered[0].id == 1
    assert ordered[-1].id == 2


def test_augment_chunks_respects_flags():
    chunks = [_chunk(1, "x :: 10h"), _chunk(2, "padding only")]
    no_compress = augment_chunks(chunks, compress=False, reorder=False)
    assert no_compress[0].content == chunks[0].content
    no_reorder = augment_chunks(chunks, compress=True, reorder=False)
    assert no_reorder[0].id == 1
