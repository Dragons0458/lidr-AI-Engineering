"""Local SQLite persistence for the Streamlit frontend (mirrors estimator-web tables)."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

_UI_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DB_PATH = str(_UI_DATA_DIR / "frontend.db")


def get_db_path() -> str:
    return os.getenv("STREAMLIT_DB_PATH", DEFAULT_DB_PATH)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(db_path: str) -> None:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _connect(db_path: str) -> sqlite3.Connection:
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource
def _cached_connection(db_path: str) -> sqlite3.Connection:
    conn = _connect(db_path)
    init_schema(conn)
    return conn


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    if db_path is not None:
        conn = _connect(path)
        init_schema(conn)
        return conn
    return _cached_connection(path)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS estimations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            project_type TEXT NOT NULL,
            detail_level TEXT NOT NULL,
            output_format TEXT NOT NULL,
            response_payload TEXT NOT NULL,
            prompt_version TEXT,
            cached INTEGER DEFAULT 0,
            chat_session_id INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            remote_session_id TEXT NOT NULL UNIQUE,
            latest_metadata TEXT,
            turn_count INTEGER DEFAULT 0,
            runtime_snapshot TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunking_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategies TEXT NOT NULL,
            queries TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            corpus_label TEXT,
            corpus_count INTEGER,
            response_payload TEXT NOT NULL,
            duration_ms INTEGER,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _parse_json_column(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def save_estimation(
    *,
    description: str,
    project_type: str,
    detail_level: str,
    output_format: str,
    response_payload: dict[str, Any],
    prompt_version: str | None = None,
    cached: bool = False,
    chat_session_id: int | None = None,
    db_path: str | None = None,
) -> int:
    conn = get_connection(db_path)
    cursor = conn.execute(
        """
        INSERT INTO estimations (
            description, project_type, detail_level, output_format,
            response_payload, prompt_version, cached, chat_session_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            description,
            project_type,
            detail_level,
            output_format,
            json.dumps(response_payload, ensure_ascii=False),
            prompt_version,
            1 if cached else 0,
            chat_session_id,
            _utc_now(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_estimations(
    *, limit: int = 20, db_path: str | None = None
) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT id, description, project_type, detail_level, output_format,
               response_payload, prompt_version, cached, chat_session_id, created_at
        FROM estimations
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = _row_to_dict(row)
        item["response_payload"] = _parse_json_column(item["response_payload"])
        results.append(item)
    return results


def get_estimation(
    estimation_id: int, *, db_path: str | None = None
) -> dict[str, Any] | None:
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT id, description, project_type, detail_level, output_format,
               response_payload, prompt_version, cached, chat_session_id, created_at
        FROM estimations
        WHERE id = ?
        """,
        (estimation_id,),
    ).fetchone()
    if row is None:
        return None
    item = _row_to_dict(row)
    item["response_payload"] = _parse_json_column(item["response_payload"])
    return item


def upsert_chat_session(
    *,
    remote_session_id: str,
    latest_metadata: dict[str, Any] | None = None,
    turn_count: int = 0,
    runtime_snapshot: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> int:
    conn = get_connection(db_path)
    now = _utc_now()
    metadata_json = json.dumps(latest_metadata or {}, ensure_ascii=False)
    snapshot_json = json.dumps(runtime_snapshot or {}, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO chat_sessions (
            remote_session_id, latest_metadata, turn_count, runtime_snapshot,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(remote_session_id) DO UPDATE SET
            latest_metadata = excluded.latest_metadata,
            turn_count = excluded.turn_count,
            runtime_snapshot = excluded.runtime_snapshot,
            updated_at = excluded.updated_at
        """,
        (remote_session_id, metadata_json, turn_count, snapshot_json, now, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM chat_sessions WHERE remote_session_id = ?",
        (remote_session_id,),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def list_chat_sessions(
    *, limit: int = 20, db_path: str | None = None
) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT id, remote_session_id, latest_metadata, turn_count,
               runtime_snapshot, created_at, updated_at
        FROM chat_sessions
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = _row_to_dict(row)
        item["latest_metadata"] = _parse_json_column(item["latest_metadata"])
        item["runtime_snapshot"] = _parse_json_column(item["runtime_snapshot"])
        results.append(item)
    return results


def get_chat_session(
    session_id: int, *, db_path: str | None = None
) -> dict[str, Any] | None:
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT id, remote_session_id, latest_metadata, turn_count,
               runtime_snapshot, created_at, updated_at
        FROM chat_sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    item = _row_to_dict(row)
    item["latest_metadata"] = _parse_json_column(item["latest_metadata"])
    item["runtime_snapshot"] = _parse_json_column(item["runtime_snapshot"])
    return item


def save_comparison(
    *,
    strategies: list[str],
    queries: list[str],
    top_k: int,
    corpus_label: str,
    corpus_count: int,
    response_payload: dict[str, Any],
    duration_ms: int,
    db_path: str | None = None,
) -> int:
    conn = get_connection(db_path)
    cursor = conn.execute(
        """
        INSERT INTO chunking_comparisons (
            strategies, queries, top_k, corpus_label, corpus_count,
            response_payload, duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            json.dumps(strategies, ensure_ascii=False),
            json.dumps(queries, ensure_ascii=False),
            top_k,
            corpus_label,
            corpus_count,
            json.dumps(response_payload, ensure_ascii=False),
            duration_ms,
            _utc_now(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_comparisons(
    *, limit: int = 20, db_path: str | None = None
) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT id, strategies, queries, top_k, corpus_label, corpus_count,
               response_payload, duration_ms, created_at
        FROM chunking_comparisons
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        item = _row_to_dict(row)
        item["strategies"] = _parse_json_column(item["strategies"])
        item["queries"] = _parse_json_column(item["queries"])
        item["response_payload"] = _parse_json_column(item["response_payload"])
        results.append(item)
    return results


def get_comparison(
    comparison_id: int, *, db_path: str | None = None
) -> dict[str, Any] | None:
    conn = get_connection(db_path)
    row = conn.execute(
        """
        SELECT id, strategies, queries, top_k, corpus_label, corpus_count,
               response_payload, duration_ms, created_at
        FROM chunking_comparisons
        WHERE id = ?
        """,
        (comparison_id,),
    ).fetchone()
    if row is None:
        return None
    item = _row_to_dict(row)
    item["strategies"] = _parse_json_column(item["strategies"])
    item["queries"] = _parse_json_column(item["queries"])
    item["response_payload"] = _parse_json_column(item["response_payload"])
    return item
