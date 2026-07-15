from __future__ import annotations

import sqlite3

import pytest

from streamlit_ui.agents import AgentProfile
from streamlit_ui.store import (
    clear_rag_run_downstream,
    confirm_rag_estimation_run,
    create_agent_profile,
    create_rag_estimation_run,
    delete_agent_profile,
    get_agent_profile,
    get_chat_session,
    get_comparison,
    get_connection,
    get_default_agent_profile,
    get_estimation,
    get_rag_estimation_run,
    init_schema,
    list_agent_profiles,
    list_chat_sessions,
    list_comparisons,
    list_estimations,
    list_rag_estimation_runs,
    save_comparison,
    save_estimation,
    seed_agent_profiles,
    set_default_agent_profile,
    update_agent_profile,
    update_rag_estimation_run,
    upsert_chat_session,
)


def test_estimation_round_trip(tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    estimation_id = save_estimation(
        description="Build a SaaS portal with OAuth",
        project_type="web_saas",
        detail_level="medium",
        output_format="phases_table",
        response_payload={
            "estimation": "Phase 1…",
            "cost_usd": 0.01,
            "cache_hit": True,
        },
        prompt_version="v1",
        cached=True,
        db_path=db_path,
    )
    rows = list_estimations(db_path=db_path)
    assert len(rows) == 1
    assert rows[0]["id"] == estimation_id
    assert rows[0]["response_payload"]["cost_usd"] == 0.01

    record = get_estimation(estimation_id, db_path=db_path)
    assert record is not None
    assert record["description"].startswith("Build a SaaS")
    assert record["cached"] == 1


def test_chat_session_upsert(tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    first_id = upsert_chat_session(
        remote_session_id="sess-abc",
        latest_metadata={"project_name": "Alpha"},
        turn_count=1,
        runtime_snapshot={"anchors_count": 0},
        db_path=db_path,
    )
    second_id = upsert_chat_session(
        remote_session_id="sess-abc",
        latest_metadata={"project_name": "Alpha v2"},
        turn_count=3,
        runtime_snapshot={"anchors_count": 2},
        db_path=db_path,
    )
    assert first_id == second_id

    sessions = list_chat_sessions(db_path=db_path)
    assert len(sessions) == 1
    assert sessions[0]["turn_count"] == 3
    assert sessions[0]["latest_metadata"]["project_name"] == "Alpha v2"

    loaded = get_chat_session(first_id, db_path=db_path)
    assert loaded is not None
    assert loaded["runtime_snapshot"]["anchors_count"] == 2


def test_estimation_linked_to_chat_session(tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    chat_id = upsert_chat_session(
        remote_session_id="sess-linked",
        turn_count=1,
        db_path=db_path,
    )
    save_estimation(
        description="Turn 1",
        project_type="web_saas",
        detail_level="medium",
        output_format="narrative",
        response_payload={"estimation": "ok"},
        chat_session_id=chat_id,
        db_path=db_path,
    )
    rows = list_estimations(db_path=db_path)
    assert rows[0]["chat_session_id"] == chat_id


def test_comparison_round_trip(tmp_path) -> None:
    db_path = str(tmp_path / "test.db")
    payload = {
        "stats_per_strategy": {
            "structural": {"ingestion_cost_usd": 0.0, "n_chunks": 10},
        },
        "queries_per_strategy": {},
    }
    comparison_id = save_comparison(
        strategies=["structural", "recursive"],
        queries=["OAuth"],
        top_k=3,
        corpus_label="budgets_sample",
        corpus_count=15,
        response_payload=payload,
        duration_ms=1200,
        db_path=db_path,
    )
    rows = list_comparisons(db_path=db_path)
    assert rows[0]["strategies"] == ["structural", "recursive"]
    assert rows[0]["queries"] == ["OAuth"]

    record = get_comparison(comparison_id, db_path=db_path)
    assert record is not None
    assert record["duration_ms"] == 1200
    assert (
        record["response_payload"]["stats_per_strategy"]["structural"]["n_chunks"] == 10
    )


def _delete_all_profiles(db_path: str) -> None:
    for profile in list_agent_profiles(db_path=db_path):
        delete_agent_profile(profile["id"], db_path=db_path)


def test_schema_migration_is_non_destructive(tmp_path) -> None:
    db_path = str(tmp_path / "migration.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE legacy_data (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO legacy_data(value) VALUES ('preserved')")
    conn.commit()

    init_schema(conn)
    init_schema(conn)

    assert conn.execute("SELECT value FROM legacy_data").fetchone()[0] == "preserved"
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agent_profiles'"
    ).fetchone()


def test_profile_presets_bootstrap_only_once(tmp_path) -> None:
    db_path = str(tmp_path / "profiles.db")
    assert [profile["name"] for profile in list_agent_profiles(db_path=db_path)] == [
        "Estándar",
        "Exhaustivo",
        "Veloz (debug)",
    ]
    default = get_default_agent_profile(db_path=db_path)
    assert default is not None and default["name"] == "Estándar"

    delete_agent_profile(default["id"], db_path=db_path)
    seed_agent_profiles(db_path=db_path)
    get_connection(db_path).close()

    assert "Estándar" not in {
        profile["name"] for profile in list_agent_profiles(db_path=db_path)
    }


def test_profile_crud_case_insensitive_uniqueness_and_atomic_default(tmp_path) -> None:
    db_path = str(tmp_path / "profiles.db")
    _delete_all_profiles(db_path)
    first_id = create_agent_profile(
        AgentProfile(
            name="Primary",
            persona="Search carefully.",
            config={"model": "gpt-5"},
            is_default=True,
        ),
        db_path=db_path,
    )
    second_id = create_agent_profile(
        name="Secondary",
        config={"max_iterations": 6},
        db_path=db_path,
    )
    with pytest.raises(sqlite3.IntegrityError):
        create_agent_profile(name="primary", db_path=db_path)

    update_agent_profile(
        second_id,
        name="Secondary updated",
        persona="Fast search.",
        config={"reasoning_effort": "low"},
        db_path=db_path,
    )
    assert get_agent_profile(second_id, db_path=db_path)["persona"] == "Fast search."

    set_default_agent_profile(second_id, db_path=db_path)
    assert get_default_agent_profile(db_path=db_path)["id"] == second_id
    assert get_agent_profile(first_id, db_path=db_path)["is_default"] is False

    with pytest.raises(KeyError):
        set_default_agent_profile(9999, db_path=db_path)
    assert get_default_agent_profile(db_path=db_path)["id"] == second_id


def test_zero_profiles_no_default_and_delete_last_profile_are_valid(tmp_path) -> None:
    db_path = str(tmp_path / "empty.db")
    _delete_all_profiles(db_path)
    assert list_agent_profiles(db_path=db_path) == []
    assert get_default_agent_profile(db_path=db_path) is None

    profile_id = create_agent_profile(name="Only", is_default=True, db_path=db_path)
    delete_agent_profile(profile_id, db_path=db_path)
    assert list_agent_profiles(db_path=db_path) == []
    assert get_default_agent_profile(db_path=db_path) is None


def test_avatar_round_trip_replace_and_delete(tmp_path) -> None:
    db_path = str(tmp_path / "avatar.db")
    profile_id = create_agent_profile(
        name="Avatar",
        avatar_filename="face.png",
        avatar_content_type="image/png",
        avatar_bytes=b"\x89PNG\r\n\x1a\ncontent",
        db_path=db_path,
    )
    loaded = get_agent_profile(profile_id, db_path=db_path)
    assert loaded["avatar_bytes"].startswith(b"\x89PNG")

    update_agent_profile(
        profile_id,
        avatar_filename="face.webp",
        avatar_content_type="image/webp",
        avatar_bytes=b"RIFF\x04\x00\x00\x00WEBPcontent",
        db_path=db_path,
    )
    assert (
        get_agent_profile(profile_id, db_path=db_path)["avatar_content_type"]
        == "image/webp"
    )
    update_agent_profile(
        profile_id,
        avatar_filename=None,
        avatar_content_type=None,
        avatar_bytes=None,
        db_path=db_path,
    )
    loaded = get_agent_profile(profile_id, db_path=db_path)
    assert loaded["avatar_filename"] is None
    assert loaded["avatar_bytes"] is None


def _full_run_values() -> dict:
    return {
        "reformulation_payload": {"query": {"project": "Portal"}},
        "structure_response": {"estimate": {"modules": []}},
        "reviewed_structure": [{"module": "Auth", "task": "OAuth"}],
        "task_hours_response": {"tasks": [{"estimated_hours": 8}]},
        "gate_report": {"passed": True},
        "final_rows": [{"task": "OAuth", "estimated_hours": 8}],
        "one_shot_result": {"estimate": "legacy"},
        "structure_profile_id": 10,
        "structure_profile_snapshot": {"name": "Structure"},
        "hours_profile_id": 11,
        "hours_profile_snapshot": {"name": "Hours"},
        "total_hours": 8.0,
        "total_engineer_days": 1.0,
        "total_cost_eur": 600.0,
        "last_error": "old error",
    }


def test_full_run_create_and_update_round_trip(tmp_path) -> None:
    db_path = str(tmp_path / "runs.db")
    run_id = create_rag_estimation_run(
        mode="agentic",
        transcript="A complete transcript",
        status="structure_review",
        current_step="structure",
        db_path=db_path,
        **_full_run_values(),
    )
    update_rag_estimation_run(
        run_id,
        mode="deterministic",
        status="hours_review",
        current_step="hours",
        task_hours_response={"tasks": [{"estimated_hours": 12}]},
        total_hours=12,
        db_path=db_path,
    )
    run = get_rag_estimation_run(run_id, db_path=db_path)
    assert run["mode"] == "deterministic"
    assert run["task_hours_response"]["tasks"][0]["estimated_hours"] == 12
    assert run["structure_profile_snapshot"] == {"name": "Structure"}
    assert run["last_error"] is None


@pytest.mark.parametrize(
    ("from_step", "cleared", "status", "current_step"),
    [
        (
            "reformulation",
            {
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
            },
            "draft",
            "transcript",
        ),
        (
            "structure",
            {
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
            },
            "draft",
            "reformulation",
        ),
        (
            "structure_review",
            {
                "reviewed_structure",
                "task_hours_response",
                "gate_report",
                "final_rows",
                "hours_profile_id",
                "hours_profile_snapshot",
                "total_hours",
                "total_engineer_days",
                "total_cost_eur",
            },
            "structure_review",
            "structure",
        ),
        (
            "hours",
            {
                "task_hours_response",
                "gate_report",
                "final_rows",
                "hours_profile_id",
                "hours_profile_snapshot",
                "total_hours",
                "total_engineer_days",
                "total_cost_eur",
            },
            "structure_review",
            "structure_review",
        ),
        (
            "gate",
            {
                "gate_report",
                "final_rows",
                "total_hours",
                "total_engineer_days",
                "total_cost_eur",
            },
            "hours_review",
            "hours",
        ),
        (
            "final_review",
            {
                "final_rows",
                "total_hours",
                "total_engineer_days",
                "total_cost_eur",
            },
            "hours_review",
            "final_review",
        ),
    ],
)
def test_clear_rag_run_downstream_full_invalidation_matrix(
    tmp_path, from_step, cleared, status, current_step
) -> None:
    db_path = str(tmp_path / f"{from_step}.db")
    values = _full_run_values()
    run_id = create_rag_estimation_run(
        mode="agentic",
        transcript="Transcript",
        status="failed",
        current_step="failed",
        db_path=db_path,
        **values,
    )
    conn = get_connection(db_path)
    conn.execute(
        "UPDATE rag_estimation_runs SET confirmed_at = 'old-confirmation' WHERE id = ?",
        (run_id,),
    )
    conn.commit()

    clear_rag_run_downstream(run_id, from_step, db_path=db_path)
    run = get_rag_estimation_run(run_id, db_path=db_path)
    assert run["status"] == status
    assert run["current_step"] == current_step
    assert run["confirmed_at"] is None
    assert run["last_error"] is None
    for key in cleared:
        assert run[key] is None, key
    for key, value in values.items():
        if key not in cleared and key != "last_error":
            assert run[key] == value, key


def test_confirmed_run_is_immutable_and_persists_totals(tmp_path) -> None:
    db_path = str(tmp_path / "confirmed.db")
    run_id = create_rag_estimation_run(
        mode="agentic", transcript="Transcript", db_path=db_path
    )
    confirm_rag_estimation_run(
        run_id,
        final_rows=[{"task": "OAuth", "estimated_hours": 8, "cost_eur": 600}],
        total_hours=8,
        total_engineer_days=1,
        total_cost_eur=600,
        db_path=db_path,
    )
    run = get_rag_estimation_run(run_id, db_path=db_path)
    assert run["status"] == "confirmed"
    assert run["confirmed_at"]
    assert (run["total_hours"], run["total_engineer_days"], run["total_cost_eur"]) == (
        8,
        1,
        600,
    )
    with pytest.raises(ValueError, match="immutable"):
        update_rag_estimation_run(run_id, transcript="changed", db_path=db_path)
    with pytest.raises(ValueError, match="immutable"):
        clear_rag_run_downstream(run_id, "hours", db_path=db_path)


def test_profile_snapshot_survives_profile_edit_and_delete(tmp_path) -> None:
    db_path = str(tmp_path / "snapshot.db")
    profile_id = create_agent_profile(
        name="Original", persona="Original persona", db_path=db_path
    )
    snapshot = get_agent_profile(profile_id, db_path=db_path)
    snapshot.pop("avatar_bytes")
    snapshot.pop("config")
    run_id = create_rag_estimation_run(
        mode="agentic",
        transcript="Transcript",
        structure_profile_id=profile_id,
        structure_profile_snapshot=snapshot,
        db_path=db_path,
    )
    update_agent_profile(profile_id, name="Renamed", db_path=db_path)
    delete_agent_profile(profile_id, db_path=db_path)

    run = get_rag_estimation_run(run_id, db_path=db_path)
    assert run["structure_profile_snapshot"]["name"] == "Original"


def test_history_filters_and_descending_order(tmp_path) -> None:
    db_path = str(tmp_path / "history.db")
    first = create_rag_estimation_run(
        mode="agentic", transcript="First", db_path=db_path
    )
    second = create_rag_estimation_run(
        mode="deterministic",
        transcript="Second",
        status="hours_review",
        db_path=db_path,
    )
    rows = list_rag_estimation_runs(db_path=db_path)
    assert [row["id"] for row in rows[:2]] == [second, first]
    assert [
        row["id"]
        for row in list_rag_estimation_runs(
            status="hours_review", mode="deterministic", db_path=db_path
        )
    ] == [second]
