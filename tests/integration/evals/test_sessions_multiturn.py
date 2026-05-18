import os

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_EVALS") != "1",
    reason="Set RUN_LLM_EVALS=1 to run multi-turn session evals against the real LLM.",
)

API_PREFIX = "/api/v1"


@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.anyio
async def test_session_continuity_keeps_project_context_across_turns() -> None:
    try:
        from deepeval import assert_test
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise AssertionError(
            "deepeval is required when RUN_LLM_EVALS=1. "
            "Install it in the dev environment before running multi-turn eval tests."
        ) from exc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        create_response = await client.post(f"{API_PREFIX}/sessions")
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        first_turn_response = await client.post(
            f"{API_PREFIX}/sessions/{session_id}/estimate",
            data={
                "description": (
                    "Project name: AtlasBoard. Build an internal admin dashboard for "
                    "a pharmacy chain with role-based permissions and stock reporting."
                )
            },
        )
        assert first_turn_response.status_code == 200
        first_turn_payload = first_turn_response.json()
        first_turn_metadata = first_turn_payload.get("project_metadata") or {}
        project_name = first_turn_metadata.get("project_name")
        assert project_name, "Expected non-empty project_name extracted in first turn."

        second_turn_response = await client.post(
            f"{API_PREFIX}/sessions/{session_id}/estimate",
            data={
                "description": (
                    "For the same project, add SOC 2 audit logging and weekly "
                    "operational reports for supervisors."
                )
            },
        )
        assert second_turn_response.status_code == 200
        second_turn_payload = second_turn_response.json()
        second_turn_metadata = second_turn_payload.get("project_metadata") or {}
        assert second_turn_metadata.get("project_name") == project_name

    continuity_metric = GEval(
        name="SessionContinuity",
        criteria=(
            "Evaluate whether the actual output remains coherent with the project "
            "established in turn 1 when answering turn 2. The output should keep "
            "the same project context and integrate the new turn-2 requirements "
            "(SOC 2 audit logging and weekly operational reports). Penalize outputs "
            "that drift to an unrelated project or ignore the carry-over context."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    test_case = LLMTestCase(
        input=(
            f"Turn 1 established project_name={project_name}. "
            "Turn 1 scope: internal admin dashboard for pharmacy chain with RBAC and "
            "stock reporting. "
            "Turn 2 request: add SOC 2 audit logging and weekly operational reports."
        ),
        actual_output=second_turn_payload["estimation"],
    )
    assert_test(test_case, [continuity_metric])
