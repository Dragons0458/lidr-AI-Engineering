"""Session 11 — HNSW halfvec indexes on all chunk collections.

Revision ID: 0005_session11_hnsw_multi_index
Revises: 0004_session10_multi_index
Create Date: 2026-07-04 00:00:00

Creates expression HNSW indexes on ``(embedding::halfvec(1536))`` with
``halfvec_cosine_ops`` for budget, transcript and technical_doc chunk tables.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0005_session11_hnsw_multi_index"
down_revision: Union[str, None] = "0004_session10_multi_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLES = ("budget_chunks", "transcript_chunks", "technical_doc_chunks")


def upgrade() -> None:
    for table in _TABLES:
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS ix_{table}_embedding_hnsw
            ON {table}
            USING hnsw ((embedding::halfvec(1536)) halfvec_cosine_ops)
            WITH (m = 16, ef_construction = 128)
            """
        )


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_embedding_hnsw")
