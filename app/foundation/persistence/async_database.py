"""Async SQLAlchemy engine and session factory for Session 8 vector persistence.

Runs in parallel with the synchronous stack in ``database.py`` (Session 6 PII/jobs).
Alembic migrations stay synchronous (+psycopg); this module derives +asyncpg from
the same ``DATABASE_URL`` so there is a single source of truth.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings


def _async_url() -> str:
    return get_settings().DATABASE_URL.replace("+psycopg", "+asyncpg")


@lru_cache
def create_async_engine_from_settings() -> AsyncEngine:
    return create_async_engine(_async_url(), pool_pre_ping=True, future=True)


AsyncSessionLocal = async_sessionmaker(
    bind=create_async_engine_from_settings(),
    autoflush=False,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session
