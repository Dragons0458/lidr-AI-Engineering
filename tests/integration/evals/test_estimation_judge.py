import os

import pytest

from tests.integration.evals.fixtures import golden_dataset
from tests.integration.evals.helpers import estimate_real

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_EVALS") != "1",
    reason="Set RUN_LLM_EVALS=1 to run judge tests against the real LLM.",
)


@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.parametrize("golden", golden_dataset.goldens)
def test_scope_coherence(golden) -> None:
    try:
        from deepeval import assert_test
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, SingleTurnParams
    except ImportError as exc:
        raise AssertionError(
            "deepeval is required when RUN_LLM_EVALS=1. "
            "Install it in the dev environment before running judge tests."
        ) from exc

    coherence_metric = GEval(
        name="ScopeCoherence",
        criteria=(
            "Evaluate whether the components, assumptions, risks, and effort "
            "breakdown in the actual output match the scope of the project described "
            "in the input. Penalize outputs that mention components or risks not "
            "implied by the input, and penalize outputs that omit major requested "
            "capabilities."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    result = estimate_real(golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=result.estimation,
    )

    assert_test(test_case, [coherence_metric])
