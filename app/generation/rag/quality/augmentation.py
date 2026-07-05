"""Deterministic content augmentation for retrieved chunks (Session 11).

Pure functions over :class:`RetrievedChunk` — no LLM calls. Compresses each chunk
to key lines (figures and structural markers) and reorders with edge-loading to
mitigate lost-in-the-middle.
"""

from __future__ import annotations

from app.generation.rag.schemas import RetrievedChunk


def extract_key_points(chunk: RetrievedChunk) -> str:
    """Keep lines with a digit or ``::``; never return empty."""
    lines = [line.strip() for line in chunk.content.splitlines() if line.strip()]
    key_lines = [
        line for line in lines if any(char.isdigit() for char in line) or "::" in line
    ]
    if key_lines:
        return "\n".join(key_lines)
    if lines:
        return lines[0]
    return chunk.content.strip() or "chunk"


def compress_chunk(chunk: RetrievedChunk) -> RetrievedChunk:
    """Shrink content while preserving ``id`` for citation verification."""
    return chunk.model_copy(update={"content": extract_key_points(chunk)})


def reorder_edge_loaded(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Edge-loading: even indices to the head, odd indices to the tail (reversed)."""
    if len(chunks) <= 2:
        return list(chunks)
    head = [chunks[i] for i in range(0, len(chunks), 2)]
    tail = [chunks[i] for i in range(1, len(chunks), 2)]
    return head + list(reversed(tail))


def augment_chunks(
    chunks: list[RetrievedChunk],
    *,
    compress: bool = True,
    reorder: bool = True,
) -> list[RetrievedChunk]:
    """Apply compress then reorder; each step is independently toggleable."""
    result = list(chunks)
    if compress:
        result = [compress_chunk(chunk) for chunk in result]
    if reorder:
        result = reorder_edge_loaded(result)
    return result
