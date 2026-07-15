from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from streamlit_ui.agent_estimation import ONE_SHOT_PATH, REFORMULATE_PATH
from streamlit_ui.store import (
    create_agent_profile,
    delete_agent_profile,
    get_rag_estimation_run,
    list_agent_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "streamlit_ui/pages/5_RAG_Estimacion.py"
TRANSCRIPT = (
    "The client needs a secure customer portal with OAuth login, account management, "
    "an audited API, deployment automation, and operational monitoring for launch."
)

STRUCTURE_RESULT = {
    "estimate": {
        "confidence": "high",
        "reasoning": "The brief is actionable.",
        "modules": [
            {
                "name": "Identity",
                "description": "Authentication",
                "tasks": [
                    {"name": "OAuth", "description": "Configure provider"},
                    {"name": "Sessions", "description": "Persist sessions"},
                ],
            }
        ],
    },
    "agent_trace": {
        "steps": [
            {
                "step": 1,
                "reasoning_summary": None,
                "tool": "propose_structure",
                "tool_args": {},
                "observation": "1 module, 2 tasks",
            }
        ]
    },
}

RESOLVED_HOURS = {
    "tasks": [
        {
            "module": "Identity",
            "task": "OAuth",
            "estimated_hours": 8,
            "has_match": True,
            "reliability": 0.8,
            "dispersion": 0.1,
            "neighbors": [{"source_id": "s1", "distance": 0.2}],
            "hours_range": None,
            "estimation_source": "deterministic",
        },
        {
            "module": "Identity",
            "task": "Sessions",
            "estimated_hours": 4,
            "has_match": True,
            "reliability": 0.9,
            "dispersion": 0.05,
            "neighbors": [{"source_id": "s2", "distance": 0.1}],
            "hours_range": None,
            "estimation_source": "deterministic",
        },
    ],
    "agent_trace": {"steps": []},
}

PARTIAL_HOURS = {
    "tasks": [
        {
            **RESOLVED_HOURS["tasks"][0],
            "estimation_source": "agent_recovery",
        },
        {
            "module": "Identity",
            "task": "Sessions",
            "estimated_hours": None,
            "has_match": False,
            "reliability": None,
            "dispersion": None,
            "neighbors": [],
            "hours_range": None,
            "estimation_source": "deterministic",
        },
    ],
    "agent_trace": {
        "steps": [
            {
                "step": 1,
                "reasoning_summary": "Search a closer analogue.",
                "tool": "search_budgets",
                "tool_args": {"query": "OAuth integration"},
                "observation": "one close match",
            },
            {
                "step": 2,
                "reasoning_summary": None,
                "tool": "derive_task_hours",
                "tool_args": {"task_ref": "task-0"},
                "observation": "8 hours",
            },
        ]
    },
}


class MockBackend:
    def __init__(self, hours_result: dict[str, Any]) -> None:
        self.hours_result = hours_result
        self.calls: list[tuple[str, Any]] = []

    def post_json(self, api_root, path, payload, **kwargs):
        self.calls.append((path, copy.deepcopy(payload)))
        if path == REFORMULATE_PATH:
            return {
                "query": {"project": "Customer portal", "sectors": ["software"]},
                "search_text": "customer portal OAuth",
            }
        if path == ONE_SHOT_PATH:
            return {"estimate": {"confidence": "high"}, "legacy": True}
        raise AssertionError(path)

    def agent_structure(self, api_root, query, profile_payload, **kwargs):
        self.calls.append(("agent_structure", copy.deepcopy(profile_payload)))
        return copy.deepcopy(STRUCTURE_RESULT)

    def deterministic_structure(self, api_root, query, **kwargs):
        self.calls.append(("deterministic_structure", copy.deepcopy(query)))
        result = copy.deepcopy(STRUCTURE_RESULT)
        result["agent_trace"] = None
        return result

    def agent_hours(self, api_root, modules, profile_payload, **kwargs):
        self.calls.append(
            ("agent_hours", {"modules": copy.deepcopy(modules), **profile_payload})
        )
        return copy.deepcopy(self.hours_result)

    def deterministic_hours(self, api_root, modules, **kwargs):
        self.calls.append(("deterministic_hours", copy.deepcopy(modules)))
        result = copy.deepcopy(RESOLVED_HOURS)
        result["agent_trace"] = None
        return result


@pytest.fixture
def wizard_factory(tmp_path, monkeypatch):
    created: list[str] = []

    def factory(hours_result: dict[str, Any] = RESOLVED_HOURS):
        db_path = str(tmp_path / f"wizard-{len(created)}.db")
        created.append(db_path)
        monkeypatch.setenv("STREAMLIT_DB_PATH", db_path)
        for profile in list_agent_profiles(db_path=db_path):
            delete_agent_profile(profile["id"], db_path=db_path)
        structure_profile = create_agent_profile(
            name="Structure specialist",
            config={"model": "gpt-5", "reasoning_effort": "medium"},
            is_default=True,
            db_path=db_path,
        )
        hours_profile = create_agent_profile(
            name="Hours specialist",
            config={"model": "gpt-5-mini", "search_top_k": 8},
            db_path=db_path,
        )
        backend = MockBackend(hours_result)
        monkeypatch.setattr(
            "streamlit_ui.common.fetch_available_agent_models",
            lambda *args, **kwargs: ["gpt-5", "gpt-5-mini"],
        )
        monkeypatch.setattr(
            "streamlit_ui.common.get_estimate_api_key", lambda: "test-key"
        )
        monkeypatch.setattr(
            "streamlit_ui.agent_estimation.post_json", backend.post_json
        )
        monkeypatch.setattr(
            "streamlit_ui.agent_estimation.post_agent_structure",
            backend.agent_structure,
        )
        monkeypatch.setattr(
            "streamlit_ui.agent_estimation.post_deterministic_structure",
            backend.deterministic_structure,
        )
        monkeypatch.setattr(
            "streamlit_ui.agent_estimation.post_agent_hours", backend.agent_hours
        )
        monkeypatch.setattr(
            "streamlit_ui.agent_estimation.post_deterministic_hours",
            backend.deterministic_hours,
        )
        app = AppTest.from_file(str(PAGE)).run(timeout=10)
        assert not app.exception
        return app, backend, db_path, structure_profile, hours_profile

    return factory


def _click(app: AppTest, label: str) -> AppTest:
    return (
        next(button for button in app.button if button.label == label)
        .click()
        .run(timeout=10)
    )


def _start_and_structure(app: AppTest) -> AppTest:
    app.text_area[0].input(TRANSCRIPT)
    app = _click(app, "Empezar")
    return _click(app, "Generar estructura")


def test_full_agentic_flow_profiles_human_gate_trace_invalidation_and_confirm(
    wizard_factory,
) -> None:
    app, backend, db_path, structure_profile, hours_profile = wizard_factory()
    app = _start_and_structure(app)
    assert not app.exception
    assert app.session_state["rag_structure_rows"][0]["task"] == "OAuth"
    assert any(
        "sin resumen de razonamiento" in str(markdown.value).lower()
        for markdown in app.markdown
    )

    hours_picker = next(
        selectbox
        for selectbox in app.selectbox
        if selectbox.label == "Perfil de recovery"
    )
    hours_picker.select(hours_profile).run()
    app = _click(app, "Estimar horas")
    assert not app.exception
    assert any("Recovery no necesario" in success.value for success in app.success)
    run_id = app.session_state["rag_run_id"]
    persisted = get_rag_estimation_run(run_id, db_path=db_path)
    assert persisted["structure_profile_id"] == structure_profile
    assert persisted["hours_profile_id"] == hours_profile
    assert persisted["status"] == "hours_review"
    assert persisted["final_rows"][0]["hourly_rate_eur"] == 75.0

    hours_call = next(
        payload for name, payload in backend.calls if name == "agent_hours"
    )
    assert [task["name"] for task in hours_call["modules"][0]["tasks"]] == [
        "OAuth",
        "Sessions",
    ]
    assert hours_call["search_top_k"] == 8

    app = _click(app, "Generar estructura")
    invalidated = get_rag_estimation_run(run_id, db_path=db_path)
    assert invalidated["task_hours_response"] is None
    assert invalidated["final_rows"] is None

    app = _click(app, "Estimar horas")
    app = _click(app, "Confirmar estimación")
    confirmed = get_rag_estimation_run(run_id, db_path=db_path)
    assert confirmed["status"] == "confirmed"
    assert confirmed["total_hours"] == 12
    assert confirmed["total_cost_eur"] == 900
    assert confirmed["confirmed_at"]


def test_deterministic_and_one_shot_flows_remain_available(wizard_factory) -> None:
    app, backend, db_path, _, _ = wizard_factory()
    app.radio[0].set_value("Determinista").run()
    app.text_area[0].input(TRANSCRIPT)
    app = _click(app, "Empezar")
    app = _click(app, "Ejecutar pipeline completo")
    app = _click(app, "Generar estructura")
    app = _click(app, "Estimar horas")

    call_names = [name for name, _ in backend.calls]
    assert ONE_SHOT_PATH in call_names
    assert "deterministic_structure" in call_names
    assert "deterministic_hours" in call_names
    run = get_rag_estimation_run(app.session_state["rag_run_id"], db_path=db_path)
    assert run["mode"] == "deterministic"
    assert run["one_shot_result"]["legacy"] is True
    assert run["task_hours_response"]["agent_trace"] is None


def test_partial_recovery_renders_trace_and_unresolved_task(wizard_factory) -> None:
    app, _, db_path, _, _ = wizard_factory(PARTIAL_HOURS)
    app = _start_and_structure(app)
    app = _click(app, "Estimar horas")

    assert any("Sin resolver" in warning.value for warning in app.warning)
    rendered = " ".join(str(markdown.value) for markdown in app.markdown)
    assert "search_budgets" in rendered
    assert "derive_task_hours" in rendered
    run = get_rag_estimation_run(app.session_state["rag_run_id"], db_path=db_path)
    assert run["final_rows"][0]["source"] == "agent_recovery"
    assert run["final_rows"][1]["estimated_hours"] is None
