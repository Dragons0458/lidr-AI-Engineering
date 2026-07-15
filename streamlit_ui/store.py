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

        CREATE TABLE IF NOT EXISTS frontend_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL COLLATE NOCASE,
            agent_type TEXT NOT NULL,
            persona TEXT NOT NULL,
            config_payload TEXT NOT NULL,
            is_default INTEGER NOT NULL DEFAULT 0,
            avatar_filename TEXT,
            avatar_content_type TEXT,
            avatar_bytes BLOB,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_profiles_type_name
            ON agent_profiles(agent_type, name COLLATE NOCASE);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_profiles_type_name_nocase
            ON agent_profiles(agent_type COLLATE NOCASE, name COLLATE NOCASE);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_profiles_one_default
            ON agent_profiles(agent_type) WHERE is_default = 1;

        CREATE TABLE IF NOT EXISTS rag_estimation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL CHECK(mode IN ('agentic', 'deterministic')),
            status TEXT NOT NULL CHECK(status IN (
                'draft', 'structure_review', 'hours_review', 'confirmed', 'failed'
            )),
            current_step TEXT NOT NULL,
            transcript TEXT NOT NULL,
            reformulation_payload TEXT,
            structure_response TEXT,
            reviewed_structure TEXT,
            task_hours_response TEXT,
            gate_report TEXT,
            final_rows TEXT,
            one_shot_result TEXT,
            structure_profile_id INTEGER,
            structure_profile_snapshot TEXT,
            hours_profile_id INTEGER,
            hours_profile_snapshot TEXT,
            total_hours REAL,
            total_engineer_days REAL,
            total_cost_eur REAL,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            confirmed_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_rag_runs_updated
            ON rag_estimation_runs(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_rag_runs_status_mode
            ON rag_estimation_runs(status, mode);
        """
    )
    conn.commit()
    _seed_agent_profiles_conn(conn)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _parse_json_column(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


_PROFILE_JSON_COLUMNS = {"config_payload"}
_RUN_JSON_COLUMNS = {
    "reformulation_payload",
    "structure_response",
    "reviewed_structure",
    "task_hours_response",
    "gate_report",
    "final_rows",
    "one_shot_result",
    "structure_profile_snapshot",
    "hours_profile_snapshot",
}
_RUN_COLUMNS = {
    "mode",
    "status",
    "current_step",
    "transcript",
    *_RUN_JSON_COLUMNS,
    "structure_profile_id",
    "hours_profile_id",
    "total_hours",
    "total_engineer_days",
    "total_cost_eur",
    "last_error",
}


def _json_dump(value: Any) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _profile_record(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = _row_to_dict(row)
    item["config_payload"] = _parse_json_column(item["config_payload"]) or {}
    item["config"] = item["config_payload"]
    item["is_default"] = bool(item["is_default"])
    return item


def _run_record(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    item = _row_to_dict(row)
    for column in _RUN_JSON_COLUMNS:
        item[column] = _parse_json_column(item.get(column))
    return item


def _seed_agent_profiles_conn(conn: sqlite3.Connection) -> None:
    marker = conn.execute(
        "SELECT 1 FROM frontend_metadata WHERE key = 'agent_profiles_seed_v1'"
    ).fetchone()
    if marker:
        return
    from streamlit_ui.agents import PRESET_PROFILES

    now = _utc_now()
    with conn:
        for profile in PRESET_PROFILES:
            conn.execute(
                """
                INSERT OR IGNORE INTO agent_profiles (
                    name, agent_type, persona, config_payload, is_default,
                    avatar_filename, avatar_content_type, avatar_bytes,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    profile.name,
                    profile.agent_type,
                    profile.persona,
                    _json_dump(profile.config),
                    int(profile.is_default),
                    now,
                    now,
                ),
            )
        conn.execute(
            """
            INSERT INTO frontend_metadata(key, value, created_at)
            VALUES ('agent_profiles_seed_v1', '1', ?)
            """,
            (now,),
        )


def seed_agent_profiles(*, db_path: str | None = None) -> None:
    _seed_agent_profiles_conn(get_connection(db_path))


def create_agent_profile(
    profile: Any | None = None,
    *,
    name: str | None = None,
    agent_type: str = "handwritten",
    persona: str = "",
    config: dict[str, Any] | None = None,
    config_payload: dict[str, Any] | None = None,
    is_default: bool = False,
    avatar_filename: str | None = None,
    avatar_content_type: str | None = None,
    avatar_bytes: bytes | None = None,
    db_path: str | None = None,
) -> int:
    from streamlit_ui.agents import AgentProfile

    candidate = profile
    if candidate is None:
        candidate = AgentProfile(
            name=name or "",
            agent_type=agent_type,
            persona=persona,
            config=config if config is not None else (config_payload or {}),
            is_default=is_default,
            avatar_filename=avatar_filename,
            avatar_content_type=avatar_content_type,
            avatar_bytes=avatar_bytes,
        )
    elif isinstance(candidate, dict):
        profile_data = dict(candidate)
        profile_data["config"] = (
            profile_data.pop("config_payload", None) or profile_data.get("config") or {}
        )
        for key in ("created_at", "updated_at"):
            profile_data.pop(key, None)
        candidate = AgentProfile(**profile_data)
    if not isinstance(candidate, AgentProfile):
        raise TypeError("profile must be an AgentProfile or mapping")
    conn = get_connection(db_path)
    now = _utc_now()
    with conn:
        if candidate.is_default:
            conn.execute(
                "UPDATE agent_profiles SET is_default = 0, updated_at = ? "
                "WHERE agent_type = ?",
                (now, candidate.agent_type),
            )
        cursor = conn.execute(
            """
            INSERT INTO agent_profiles (
                name, agent_type, persona, config_payload, is_default,
                avatar_filename, avatar_content_type, avatar_bytes,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate.name,
                candidate.agent_type,
                candidate.persona,
                _json_dump(candidate.config),
                int(candidate.is_default),
                candidate.avatar_filename,
                candidate.avatar_content_type,
                candidate.avatar_bytes,
                now,
                now,
            ),
        )
    return int(cursor.lastrowid)


def update_agent_profile(
    profile_id: int,
    profile: Any | None = None,
    *,
    db_path: str | None = None,
    **changes: Any,
) -> None:
    from streamlit_ui.agents import AgentProfile

    current = get_agent_profile(profile_id, db_path=db_path)
    if current is None:
        raise KeyError(f"Agent profile {profile_id} not found")
    if profile is not None:
        if isinstance(profile, AgentProfile):
            changes = profile.snapshot(include_avatar=True)
        elif isinstance(profile, dict):
            changes = dict(profile)
        else:
            raise TypeError("profile must be an AgentProfile or mapping")
    merged = {
        **current,
        **changes,
        "config": changes.get(
            "config", changes.get("config_payload", current["config_payload"])
        ),
    }
    candidate = AgentProfile(
        id=profile_id,
        name=merged["name"],
        agent_type=merged["agent_type"],
        persona=merged.get("persona") or "",
        config=merged["config"],
        is_default=bool(merged.get("is_default")),
        avatar_filename=merged.get("avatar_filename"),
        avatar_content_type=merged.get("avatar_content_type"),
        avatar_bytes=merged.get("avatar_bytes"),
        created_at=merged.get("created_at"),
        updated_at=merged.get("updated_at"),
    )
    conn = get_connection(db_path)
    now = _utc_now()
    with conn:
        if candidate.is_default:
            conn.execute(
                "UPDATE agent_profiles SET is_default = 0, updated_at = ? "
                "WHERE agent_type = ? AND id <> ?",
                (now, candidate.agent_type, profile_id),
            )
        conn.execute(
            """
            UPDATE agent_profiles SET
                name = ?, agent_type = ?, persona = ?, config_payload = ?,
                is_default = ?, avatar_filename = ?, avatar_content_type = ?,
                avatar_bytes = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                candidate.name,
                candidate.agent_type,
                candidate.persona,
                _json_dump(candidate.config),
                int(candidate.is_default),
                candidate.avatar_filename,
                candidate.avatar_content_type,
                candidate.avatar_bytes,
                now,
                profile_id,
            ),
        )


def delete_agent_profile(profile_id: int, *, db_path: str | None = None) -> None:
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM agent_profiles WHERE id = ?", (profile_id,))


def get_agent_profile(
    profile_id: int, *, db_path: str | None = None
) -> dict[str, Any] | None:
    row = (
        get_connection(db_path)
        .execute("SELECT * FROM agent_profiles WHERE id = ?", (profile_id,))
        .fetchone()
    )
    return _profile_record(row)


def list_agent_profiles(
    *, agent_type: str | None = "handwritten", db_path: str | None = None
) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    if agent_type is None:
        rows = conn.execute(
            "SELECT * FROM agent_profiles ORDER BY is_default DESC, name COLLATE NOCASE"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM agent_profiles WHERE agent_type = ? "
            "ORDER BY is_default DESC, name COLLATE NOCASE",
            (agent_type,),
        ).fetchall()
    return [record for row in rows if (record := _profile_record(row)) is not None]


def get_default_agent_profile(
    agent_type: str = "handwritten", *, db_path: str | None = None
) -> dict[str, Any] | None:
    row = (
        get_connection(db_path)
        .execute(
            "SELECT * FROM agent_profiles WHERE agent_type = ? AND is_default = 1",
            (agent_type,),
        )
        .fetchone()
    )
    return _profile_record(row)


def set_default_agent_profile(
    profile_id: int | None,
    *,
    agent_type: str = "handwritten",
    db_path: str | None = None,
) -> None:
    conn = get_connection(db_path)
    now = _utc_now()
    with conn:
        conn.execute(
            "UPDATE agent_profiles SET is_default = 0, updated_at = ? "
            "WHERE agent_type = ?",
            (now, agent_type),
        )
        if profile_id is not None:
            cursor = conn.execute(
                "UPDATE agent_profiles SET is_default = 1, updated_at = ? "
                "WHERE id = ? AND agent_type = ?",
                (now, profile_id, agent_type),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Agent profile {profile_id} not found")


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


def create_rag_estimation_run(
    *,
    mode: str,
    transcript: str,
    status: str = "draft",
    current_step: str = "transcript",
    db_path: str | None = None,
    **values: Any,
) -> int:
    if mode not in {"agentic", "deterministic"}:
        raise ValueError("mode must be agentic or deterministic")
    now = _utc_now()
    payload = {
        "mode": mode,
        "status": status,
        "current_step": current_step,
        "transcript": transcript,
        **{key: value for key, value in values.items() if key in _RUN_COLUMNS},
    }
    columns = list(payload)
    serialized = [
        _json_dump(payload[column]) if column in _RUN_JSON_COLUMNS else payload[column]
        for column in columns
    ]
    columns.extend(["created_at", "updated_at"])
    serialized.extend([now, now])
    placeholders = ", ".join("?" for _ in columns)
    cursor = get_connection(db_path).execute(
        f"INSERT INTO rag_estimation_runs ({', '.join(columns)}) VALUES ({placeholders})",
        serialized,
    )
    cursor.connection.commit()
    return int(cursor.lastrowid)


def update_rag_estimation_run(
    run_id: int, *, db_path: str | None = None, **changes: Any
) -> None:
    conn = get_connection(db_path)
    current = conn.execute(
        "SELECT status FROM rag_estimation_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if current is None:
        raise KeyError(f"RAG estimation run {run_id} not found")
    if current["status"] == "confirmed":
        raise ValueError("Confirmed RAG estimation runs are immutable.")
    updates = {key: value for key, value in changes.items() if key in _RUN_COLUMNS}
    if not updates:
        return
    updates.setdefault("last_error", None)
    assignments = [f"{key} = ?" for key in updates]
    values = [
        _json_dump(value) if key in _RUN_JSON_COLUMNS else value
        for key, value in updates.items()
    ]
    assignments.append("updated_at = ?")
    values.extend([_utc_now(), run_id])
    with conn:
        conn.execute(
            f"UPDATE rag_estimation_runs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )


_INVALIDATION_MATRIX: dict[str, tuple[list[str], str, str]] = {
    "reformulation": (
        [
            "reformulation_payload",
            "structure_response",
            "reviewed_structure",
            "task_hours_response",
            "gate_report",
            "final_rows",
            "one_shot_result",
            "structure_profile_id",
            "structure_profile_snapshot",
            "hours_profile_id",
            "hours_profile_snapshot",
            "total_hours",
            "total_engineer_days",
            "total_cost_eur",
        ],
        "draft",
        "transcript",
    ),
    "structure": (
        [
            "structure_response",
            "reviewed_structure",
            "task_hours_response",
            "gate_report",
            "final_rows",
            "structure_profile_id",
            "structure_profile_snapshot",
            "hours_profile_id",
            "hours_profile_snapshot",
            "total_hours",
            "total_engineer_days",
            "total_cost_eur",
        ],
        "draft",
        "reformulation",
    ),
    "structure_review": (
        [
            "reviewed_structure",
            "task_hours_response",
            "gate_report",
            "final_rows",
            "hours_profile_id",
            "hours_profile_snapshot",
            "total_hours",
            "total_engineer_days",
            "total_cost_eur",
        ],
        "structure_review",
        "structure",
    ),
    "hours": (
        [
            "task_hours_response",
            "gate_report",
            "final_rows",
            "hours_profile_id",
            "hours_profile_snapshot",
            "total_hours",
            "total_engineer_days",
            "total_cost_eur",
        ],
        "structure_review",
        "structure_review",
    ),
    "gate": (
        [
            "gate_report",
            "final_rows",
            "total_hours",
            "total_engineer_days",
            "total_cost_eur",
        ],
        "hours_review",
        "hours",
    ),
    "final_review": (
        ["final_rows", "total_hours", "total_engineer_days", "total_cost_eur"],
        "hours_review",
        "final_review",
    ),
}


def clear_rag_run_downstream(
    run_id: int, from_step: str, *, db_path: str | None = None
) -> None:
    if from_step not in _INVALIDATION_MATRIX:
        raise ValueError(f"Unknown RAG step: {from_step}")
    columns, status, current_step = _INVALIDATION_MATRIX[from_step]
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT status FROM rag_estimation_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"RAG estimation run {run_id} not found")
    if row["status"] == "confirmed":
        raise ValueError("Confirmed RAG estimation runs are immutable.")
    assignments = [f"{column} = NULL" for column in columns]
    assignments.extend(
        [
            "status = ?",
            "current_step = ?",
            "confirmed_at = NULL",
            "last_error = NULL",
            "updated_at = ?",
        ]
    )
    with conn:
        conn.execute(
            f"UPDATE rag_estimation_runs SET {', '.join(assignments)} WHERE id = ?",
            (status, current_step, _utc_now(), run_id),
        )


def confirm_rag_estimation_run(
    run_id: int,
    *,
    final_rows: list[dict[str, Any]],
    total_hours: float,
    total_engineer_days: float,
    total_cost_eur: float,
    structure_profile_snapshot: dict[str, Any] | None = None,
    hours_profile_snapshot: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> None:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT status FROM rag_estimation_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"RAG estimation run {run_id} not found")
    if row["status"] == "confirmed":
        raise ValueError("Confirmed RAG estimation runs are immutable.")
    now = _utc_now()
    with conn:
        conn.execute(
            """
            UPDATE rag_estimation_runs SET
                final_rows = ?, structure_profile_snapshot = COALESCE(?, structure_profile_snapshot),
                hours_profile_snapshot = COALESCE(?, hours_profile_snapshot),
                total_hours = ?, total_engineer_days = ?, total_cost_eur = ?,
                status = 'confirmed', current_step = 'confirmed',
                confirmed_at = ?, updated_at = ?, last_error = NULL
            WHERE id = ?
            """,
            (
                _json_dump(final_rows),
                _json_dump(structure_profile_snapshot),
                _json_dump(hours_profile_snapshot),
                total_hours,
                total_engineer_days,
                total_cost_eur,
                now,
                now,
                run_id,
            ),
        )


def get_rag_estimation_run(
    run_id: int, *, db_path: str | None = None
) -> dict[str, Any] | None:
    row = (
        get_connection(db_path)
        .execute("SELECT * FROM rag_estimation_runs WHERE id = ?", (run_id,))
        .fetchone()
    )
    return _run_record(row)


def list_rag_estimation_runs(
    *,
    status: str | None = None,
    mode: str | None = None,
    limit: int = 100,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    values: list[Any] = []
    if status:
        clauses.append("status = ?")
        values.append(status)
    if mode:
        clauses.append("mode = ?")
        values.append(mode)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    values.append(limit)
    rows = (
        get_connection(db_path)
        .execute(
            f"SELECT * FROM rag_estimation_runs {where} "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            values,
        )
        .fetchall()
    )
    return [record for row in rows if (record := _run_record(row)) is not None]


def clone_rag_estimation_run(run_id: int, *, db_path: str | None = None) -> int:
    source = get_rag_estimation_run(run_id, db_path=db_path)
    if source is None:
        raise KeyError(f"RAG estimation run {run_id} not found")
    return create_rag_estimation_run(
        mode=source["mode"],
        transcript=source["transcript"],
        reformulation_payload=source["reformulation_payload"],
        status="draft",
        current_step="reformulation"
        if source["reformulation_payload"]
        else "transcript",
        db_path=db_path,
    )
