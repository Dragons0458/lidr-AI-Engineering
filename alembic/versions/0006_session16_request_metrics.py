"""Session 16 — per-request metrics table for production observability.

Revision ID: 0006_session16_request_metrics
Revises: 0005_session11_hnsw_multi_index
Create Date: 2026-08-21 00:00:00

Stores one row per instrumented estimate request: latency, tokens, cost,
status, confidence. Never transcripts, prompts, completions, or API keys.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_session16_request_metrics"
down_revision: Union[str, None] = "0005_session11_hnsw_multi_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "request_metrics",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.Text, nullable=False),
        sa.Column("route", sa.Text, nullable=False),
        sa.Column("http_status", sa.Integer, nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Float, nullable=False),
        sa.Column("llm_calls", sa.Integer, nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("model", sa.Text, nullable=True),
        sa.Column("confidence", sa.String(length=32), nullable=True),
        sa.Column("abstained", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("grounded_ratio", sa.Float, nullable=True),
        sa.Column("cache_hit", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_request_metrics_created_at",
        "request_metrics",
        ["created_at"],
    )
    op.create_index(
        "idx_request_metrics_request_id",
        "request_metrics",
        ["request_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_request_metrics_request_id", table_name="request_metrics")
    op.drop_index("idx_request_metrics_created_at", table_name="request_metrics")
    op.drop_table("request_metrics")
