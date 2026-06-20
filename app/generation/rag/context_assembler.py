"""Augmentation: assemble retrieved chunks into a delimited context block."""

from __future__ import annotations

from app.generation.rag.schemas import RetrievedChunk


def _wrap_chunk(chunk: RetrievedChunk) -> str:
    """Render a single chunk as a self-describing ``<source>`` XML element."""
    return (
        f'<source id="{chunk.id}" sector="{chunk.sector}" '
        f'project_year="{chunk.project_year}" chunk_type="{chunk.chunk_type}" '
        f'distance="{chunk.distance:.4f}">\n'
        f"{chunk.content}\n"
        f"</source>"
    )


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Build the XML context block fed to the generator."""
    return "\n".join(_wrap_chunk(chunk) for chunk in chunks)


def truncate_to_token_budget(
    chunks: list[RetrievedChunk],
    max_context_tokens: int,
    encoder,
) -> list[RetrievedChunk]:
    """Keep as many leading chunks as fit within ``max_context_tokens``."""
    kept: list[RetrievedChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(encoder.encode(_wrap_chunk(chunk)))
        if used + cost > max_context_tokens:
            break
        kept.append(chunk)
        used += cost
    return kept
