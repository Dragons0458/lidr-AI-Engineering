"""Integration smoke test for the Session 14 Supervisor Streamlit page."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "streamlit_ui/pages/10_Supervisor.py"


@pytest.fixture
def supervisor_app(tmp_path, monkeypatch) -> AppTest:
    db_path = str(tmp_path / "supervisor-page.db")
    monkeypatch.setenv("STREAMLIT_DB_PATH", db_path)
    monkeypatch.setattr(
        "streamlit_ui.common.get_api_root_url",
        lambda: "http://testserver",
    )
    monkeypatch.setattr(
        "streamlit_ui.common.get_estimate_api_key",
        lambda: "test-key",
    )
    return AppTest.from_file(str(PAGE)).run(timeout=15)


def test_supervisor_page_renders_with_paused_competition_state(supervisor_app) -> None:
    app = supervisor_app
    assert not app.exception

    app.session_state["supervisor_estimation_id"] = "st-s14-demo"
    app.session_state["supervisor_state"] = {
        "estimation_id": "st-s14-demo",
        "state": "paused",
        "status": "awaiting_human_review",
        "confidence": 0.35,
        "persist_requested": True,
        "pending_review": {
            "reasons": ["high_divergence", "irreversible_write_pending"],
            "confidence": 0.35,
            "threshold": 0.6,
            "divergence": {"ratio": 1.0, "level": "high", "spread": 300},
            "persist_requested": True,
            "estimate": {
                "total_hours": 200,
                "range": {"low": 100, "high": 400},
                "components": [
                    {"name": "API", "estimated_hours": 120},
                    {"name": "Auth", "estimated_hours": 80},
                ],
                "open_questions": ["Is scope closed?"],
            },
        },
        "divergence": {"ratio": 1.0, "level": "high", "spread": 300},
        "synthesis": {
            "low": 100,
            "high": 400,
            "open_questions": ["Is scope closed?"],
        },
        "proposals": [
            {"stance": "conservative", "total_hours": 400, "risks": ["integration"]},
            {"stance": "aggressive", "total_hours": 100, "risks": ["reuse"]},
        ],
        "estimate": {
            "total_hours": 200,
            "range": {"low": 100, "high": 400},
            "components": [
                {"name": "API", "estimated_hours": 120},
                {"name": "Auth", "estimated_hours": 80},
            ],
        },
        "agent_contributions": [
            {
                "agent": "persistence_agent",
                "tool": "save_estimate",
                "outcome": "deferred",
                "action": "tool:save_estimate",
                "summary": "awaiting approval",
            }
        ],
    }
    app = app.run(timeout=15)
    assert not app.exception
    # Smoke: page re-ran with injected paused/competition state.
    assert app.session_state["supervisor_estimation_id"] == "st-s14-demo"
    assert (
        app.session_state["supervisor_state"]["pending_review"]["persist_requested"]
        is True
    )
