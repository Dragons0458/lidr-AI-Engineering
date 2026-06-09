from __future__ import annotations

from streamlit_ui.store import (
    get_chat_session,
    get_comparison,
    get_estimation,
    list_chat_sessions,
    list_comparisons,
    list_estimations,
    save_comparison,
    save_estimation,
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
