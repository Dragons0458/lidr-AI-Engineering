"""Schema-level tests for the Session 8/10 ORM models."""

from __future__ import annotations

from pgvector.sqlalchemy import Vector

from app.generation.rag.store.models import (
    BudgetChunkRow,
    ChunkRow,
    DocumentRow,
    TechnicalDocChunkRow,
    TranscriptChunkRow,
)


def test_metadata_attribute_maps_to_metadata_column():
    assert DocumentRow.metadata_.expression.name == "metadata"
    assert BudgetChunkRow.metadata_.expression.name == "metadata"


def test_embedding_is_a_nullable_1536_dim_vector():
    embedding = BudgetChunkRow.__table__.c.embedding
    assert isinstance(embedding.type, Vector)
    assert embedding.type.dim == 1536
    assert embedding.nullable is True


def test_chunks_fk_cascades_on_document_delete():
    fk = next(iter(BudgetChunkRow.__table__.c.document_id.foreign_keys))
    assert fk.column.table.name == "documents"
    assert fk.ondelete == "CASCADE"


def test_no_vector_index_yet():
    index_columns = {
        col.name for index in BudgetChunkRow.__table__.indexes for col in index.columns
    }
    assert "embedding" not in index_columns


def test_budget_relational_indexes_present():
    index_names = {index.name for index in BudgetChunkRow.__table__.indexes}
    assert index_names == {
        "ix_budget_chunks_document_id",
        "ix_budget_chunks_chunk_type",
        "ix_budget_chunks_metadata_gin",
        "ix_budget_chunks_content_tsv",
    }
    assert {index.name for index in DocumentRow.__table__.indexes} == {
        "ix_documents_source_path"
    }


def test_three_collection_models_and_alias():
    assert BudgetChunkRow.__tablename__ == "budget_chunks"
    assert TranscriptChunkRow.__tablename__ == "transcript_chunks"
    assert TechnicalDocChunkRow.__tablename__ == "technical_doc_chunks"
    assert ChunkRow is BudgetChunkRow
