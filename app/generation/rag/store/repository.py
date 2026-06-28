"""Async data-access layer for the vector store.

The store never opens or commits sessions: the caller (ingest service,
retriever) owns the ``AsyncSession`` so a whole ingest — duplicate check,
document row, chunk rows — fits in ONE transaction. A failure anywhere rolls
everything back and leaves no orphan ``documents`` row.

Session 10 makes every search/persist method **collection-aware**: a ``model``
argument selects which chunk table (``budget_chunks`` / ``transcript_chunks`` /
``technical_doc_chunks``) the query runs against. It defaults to the budgets
table so all Session 8/9 callers are unaffected.
"""

from __future__ import annotations

from sqlalchemy import Integer, Row, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.models import BudgetChunkRow, DocumentRow, FTS_REGCONFIG

# The structural chunker emits one chunk per budget component; the vocabulary
# is queryable thanks to the index on ``chunk_type`` (live-session filters).
BUDGET_COMPONENT = "budget_component"


class ChunkStore:
    """CRUD + similarity search over ``documents`` and the chunk tables."""

    @staticmethod
    def _structural_filters(
        model: type,
        *,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[ColumnElement]:
        """Shared metadata filters for vector and lexical branches."""
        sector_col = model.metadata_["client_sector"].astext
        year_col = cast(model.metadata_["year"].astext, Integer)
        structural: list[ColumnElement] = []
        if sectors:
            structural.append(sector_col.in_(sectors))
        if project_year_min is not None:
            structural.append(year_col >= project_year_min)
        if project_year_max is not None:
            structural.append(year_col <= project_year_max)
        if chunk_types:
            structural.append(model.chunk_type.in_(chunk_types))
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
        chunk_type: str = BUDGET_COMPONENT,
        model: type = BudgetChunkRow,
    ) -> int:
        """Insert the document row plus all its chunk rows. No commit here —
        the caller's transaction decides when (and whether) anything lands."""
        document = DocumentRow(
            source_path=source_path,
            document_type=document_type,
            metadata_=doc_metadata,
        )
        session.add(document)
        await session.flush()

        session.add_all(
            model(
                document_id=document.id,
                chunk_type=chunk_type,
                content=chunk.text,
                embedding=chunk.embedding,
                metadata_=chunk.metadata,
            )
            for chunk in embedded_chunks
        )
        return document.id

    async def search(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        k: int,
        model: type = BudgetChunkRow,
    ) -> list[Row]:
        """k nearest chunks by cosine distance (``<=>``), sequential scan."""
        distance = model.embedding.cosine_distance(query_vector)
        stmt = (
            select(
                model.id,
                model.document_id,
                model.chunk_type,
                model.content,
                model.metadata_,
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
        model: type = BudgetChunkRow,
        extra_filters: list | None = None,
    ) -> tuple[list[Row], int]:
        """Metadata-filtered k-NN with cosine distance threshold (Session 9)."""
        structural = self._structural_filters(
            model,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        structural.extend(extra_filters or [])

        distance = model.embedding.cosine_distance(query_vector)

        count_stmt = select(func.count()).select_from(model)
        if structural:
            count_stmt = count_stmt.where(*structural)
        candidates = int((await session.execute(count_stmt)).scalar_one())

        stmt = select(
            model.id,
            model.document_id,
            model.chunk_type,
            model.content,
            model.metadata_,
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
        model: type = BudgetChunkRow,
        extra_filters: list | None = None,
    ) -> list[Row]:
        """Keyword (full-text) ranking over the ``content_tsv`` column (Session 10).

        Uses ``websearch_to_tsquery`` + ``ts_rank`` (estimador-cag signature).
        """
        tsquery = func.websearch_to_tsquery(FTS_REGCONFIG, query_text)
        rank = func.ts_rank(model.content_tsv, tsquery)

        structural_filters = self._structural_filters(
            model,
            sectors=sectors,
            project_year_min=project_year_min,
            project_year_max=project_year_max,
            chunk_types=chunk_types,
        )
        structural_filters.extend(extra_filters or [])

        stmt = (
            select(
                model.id,
                model.document_id,
                model.chunk_type,
                model.content,
                model.metadata_,
                rank.label("rank"),
            )
            .where(model.content_tsv.op("@@")(tsquery))
            .where(*structural_filters)
            .order_by(rank.desc())
            .limit(top_k)
        )
        return list((await session.execute(stmt)).all())
