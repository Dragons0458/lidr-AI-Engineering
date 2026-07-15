"""Unit tests for LangGraph Postgres URL helpers."""

from __future__ import annotations

import pytest

from app.foundation.persistence.langgraph import to_libpq_conninfo


def test_to_libpq_strips_sqlalchemy_driver():
    assert (
        to_libpq_conninfo(
            "postgresql+psycopg://estimator:estimator@localhost:5433/estimator"
        )
        == "postgresql://estimator:estimator@localhost:5433/estimator"
    )


def test_to_libpq_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        to_libpq_conninfo("mysql://localhost/db")
