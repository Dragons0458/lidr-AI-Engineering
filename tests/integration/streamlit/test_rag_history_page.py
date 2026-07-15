from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from streamlit_ui.store import (
    confirm_rag_estimation_run,
    create_agent_profile,
    create_rag_estimation_run,
    delete_agent_profile,
    get_agent_profile,
    update_rag_estimation_run,
)

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "streamlit_ui/pages/8_Historial_RAG.py"


@pytest.fixture
def history_app(tmp_path, monkeypatch):
    db_path = str(tmp_path / "history-page.db")
    monkeypatch.setenv("STREAMLIT_DB_PATH", db_path)
    profile_id = create_agent_profile(
        name="Historical profile",
        persona="Search each task.",
        config={"model": "gpt-5", "search_top_k": 8},
        db_path=db_path,
    )
    snapshot = get_agent_profile(profile_id, db_path=db_path)
    snapshot.pop("avatar_bytes")
    snapshot.pop("config")
    incomplete_id = create_rag_estimation_run(
        mode="agentic",
        transcript="Incomplete transcript",
        status="hours_review",
        current_step="hours",
        reformulation_payload={"query": {"project": "Portal"}},
        structure_response={"estimate": {"modules": [{"name": "Auth"}]}},
        reviewed_structure=[{"module": "Auth", "task": "OAuth"}],
        task_hours_response={
            "tasks": [{"task": "OAuth", "estimated_hours": 8}],
            "agent_trace": {"steps": [{"tool": "search_budgets"}]},
        },
        gate_report={"passed": True},
        final_rows=[{"task": "OAuth", "estimated_hours": 8}],
        one_shot_result={"legacy": True},
        structure_profile_id=profile_id,
        structure_profile_snapshot=snapshot,
        hours_profile_id=profile_id,
        hours_profile_snapshot=snapshot,
        total_hours=8,
        total_engineer_days=1,
        total_cost_eur=600,
        db_path=db_path,
    )
    update_rag_estimation_run(
        incomplete_id,
        status="failed",
        current_step="gate",
        last_error="Judge unavailable",
        db_path=db_path,
    )
    confirmed_id = create_rag_estimation_run(
        mode="deterministic",
        transcript="Confirmed transcript",
        structure_profile_snapshot=snapshot,
        hours_profile_snapshot=snapshot,
        db_path=db_path,
    )
    confirm_rag_estimation_run(
        confirmed_id,
        final_rows=[{"task": "API", "estimated_hours": 4, "cost_eur": 300}],
        total_hours=4,
        total_engineer_days=0.5,
        total_cost_eur=300,
        db_path=db_path,
    )
    delete_agent_profile(profile_id, db_path=db_path)
    switched: list[str] = []
    monkeypatch.setattr("streamlit.switch_page", switched.append)
    app = AppTest.from_file(str(PAGE)).run(timeout=10)
    assert not app.exception
    return app, db_path, incomplete_id, confirmed_id, switched


def test_history_lists_orders_and_filters_runs(history_app) -> None:
    app, _, incomplete_id, confirmed_id, _ = history_app
    summary = app.dataframe[0].value
    assert list(summary["id"]) == [confirmed_id, incomplete_id]

    app.selectbox[0].select("confirmed").run()
    assert list(app.dataframe[0].value["id"]) == [confirmed_id]
    app.selectbox[0].select("Todos").run()
    app.selectbox[1].select("agentic").run()
    assert list(app.dataframe[0].value["id"]) == [incomplete_id]


def test_history_shows_full_detail_and_deleted_profile_snapshots(history_app) -> None:
    app, _, incomplete_id, _, _ = history_app
    app.selectbox[2].select(incomplete_id).run()

    expander_labels = {expander.label for expander in app.expander}
    assert {
        "Transcript",
        "Reformulación",
        "Estructura propuesta",
        "Estructura revisada",
        "Task-hours y traza de recovery",
        "Hallucination gate",
        "Breakdown final",
        "One-shot",
        "Snapshot perfil de estructura",
        "Snapshot perfil de horas",
    } <= expander_labels
    rendered = " ".join(
        [str(item.value) for item in app.json]
        + [str(item.value) for item in app.markdown]
        + [str(item.value) for item in app.error]
    )
    assert "Historical profile" in rendered
    assert "Judge unavailable" in rendered


def test_incomplete_run_restores_session_and_confirmed_is_read_only(
    history_app,
) -> None:
    app, _, incomplete_id, confirmed_id, switched = history_app
    app.selectbox[2].select(incomplete_id).run()
    next(
        button for button in app.button if button.label == "Restaurar y continuar"
    ).click().run()

    assert app.session_state["rag_run_id"] == incomplete_id
    assert app.session_state["rag_transcript"] == "Incomplete transcript"
    assert app.session_state["rag_structure_rows"] == [
        {"module": "Auth", "task": "OAuth"}
    ]
    assert switched == ["pages/5_RAG_Estimacion.py"]

    app = AppTest.from_file(str(PAGE)).run(timeout=10)
    app.selectbox[2].select(confirmed_id).run()
    assert any("solo lectura" in info.value for info in app.info)
    assert all(button.label != "Restaurar y continuar" for button in app.button)
