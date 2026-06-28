"""Async data-access layer for the vector store.

The store never opens or commits sessions: the caller (ingest service,
retriever) owns the ``AsyncSession`` so a whole ingest — duplicate check,
document row, chunk rows — fits in ONE transaction. A failure anywhere rolls
everything back and leaves no orphan ``documents`` row.
"""

from __future__ import annotations

from sqlalchemy import Integer, Row, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.models import ChunkRow, DocumentRow

# The structural chunker emits one chunk per budget component; the vocabulary
# is queryable thanks to the index on ``chunk_type`` (live-session filters).
BUDGET_COMPONENT = "budget_component"


class ChunkStore:
    """CRUD + similarity search over ``documents``/``chunks``."""

    def _structural_filters(
        self,
        *,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[ColumnElement]:
        """Shared metadata filters for vector and lexical branches."""
        sector_col = ChunkRow.metadata_["client_sector"].astext
        year_col = cast(ChunkRow.metadata_["year"].astext, Integer)
        structural: list[ColumnElement] = []
        if sectors:
            structural.append(sector_col.in_(sectors))
        if project_year_min is not None:
            structural.append(year_col >= project_year_min)
        if project_year_max is not None:
            structural.append(year_col <= project_year_max)
        if chunk_types:
            structural.append(ChunkRow.chunk_type.in_(chunk_types))
        return structural

    async def find_document_id(
        self, session: AsyncSession, source_path: str
    ) -> int | None:
        """Return the id of the document already ingested from ``source_path``,
        or ``None``. Backs the application-level 409 duplicate guard."""
        stmt = select(DocumentRow.id).where(DocumentRow.source_path == source_path)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def persist_document_with_chunks(
        self,
        session: AsyncSession,
        *,
        source_path: str,
        document_type: str,
        doc_metadata: dict,
        embedded_chunks: list[EmbeddedChunk],
    ) -> int:
        """Insert the document row plus all its chunk rows. No commit here —
        the caller's transaction decides when (and whether) anything lands."""
        document = DocumentRow(
            source_path=source_path,
            document_type=document_type,
            metadata_=doc_metadata,
        )
        session.add(document)
        await session.flush()  # assigns document.id without committing

        session.add_all(
            ChunkRow(
                document_id=document.id,
                chunk_type=BUDGET_COMPONENT,
                content=chunk.text,
                embedding=chunk.embedding,
                metadata_=chunk.metadata,
            )
            for chunk in embedded_chunks
        )
        return document.id

    async def search(
        self, session: AsyncSession, *, query_vector: list[float], k: int
    ) -> list[Row]:
        """k nearest chunks by cosine distance (``<=>``), sequential scan.

        Cosine over L2/inner product: OpenAI embeddings are normalized so the
        ranking would be equivalent, but cosine keeps us aligned with the RAG
        literature AND with the ``vector_cosine_ops`` operator class of the
        HNSW index the live session adds — operator/index mismatch makes
        Postgres silently ignore the index.
        """
        distance = ChunkRow.embedding.cosine_distance(query_vector)
        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(k)
        )
        return list((await session.execute(stmt)).all())

    async def search_filtered(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> tuple[list[Row], int]:
        """Metadata-filtered k-NN with cosine distance threshold (Session 9)."""
        structural = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        distance = ChunkRow.embedding.cosine_distance(query_vector)
        count_stmt = select(func.count()).select_from(ChunkRow)
        if structural:
            count_stmt = count_stmt.where(*structural)
        candidates = int((await session.execute(count_stmt)).scalar_one())
        stmt = select(
            ChunkRow.id,
            ChunkRow.document_id,
            ChunkRow.chunk_type,
            ChunkRow.content,
            ChunkRow.metadata_,
            distance.label("distance"),
        )
        if structural:
            stmt = stmt.where(*structural)
        stmt = (
            stmt.where(distance <= distance_threshold).order_by(distance).limit(top_k)
        )
        rows = list((await session.execute(stmt)).all())
        return rows, candidates

    async def search_lexical(
        self,
        session: AsyncSession,
        *,
        query_text: str,
        top_k: int = 50,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[Row]:
        """Keyword (full-text) ranking over the ``content_tsv`` column (Session 10).

        The lexical branch of hybrid search: ``websearch_to_tsquery`` tolerates
        natural-language input and search-style syntax better than ``plainto_tsquery``,
        which AND-every lexeme and often returns zero rows on long queries. ``@@``
        keeps only chunks that match; ``ts_rank`` scores them (higher is better,
        opposite to vector distance). The ``english`` config MUST match the generated
        column's config (migration 0003) or the GIN index is bypassed and matching
        silently changes. Structural filters mirror ``search_filtered`` so the two
        branches see the same candidate space.

        Returns rows ordered by rank DESC (most relevant first), capped at
        ``top_k``. ``rank`` rides along for debugging; fusion only uses the ordering.
        """
        tsquery = func.websearch_to_tsquery("english", query_text)
        rank = func.ts_rank(ChunkRow.content_tsv, tsquery)

        structural_filters = self._structural_filters(
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )

        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                rank.label("rank"),
            )
            .where(ChunkRow.content_tsv.op("@@")(tsquery))
            .where(*structural_filters)
            .order_by(rank.desc())
            .limit(top_k)
        )
        return list((await session.execute(stmt)).all())
