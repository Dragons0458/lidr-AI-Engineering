import os
import statistics

import pytest

from app.schemas.estimation import EstimationResponse
from tests.integration.evals.fixtures import golden_dataset
from tests.integration.evals.helpers import (
    estimate_real,
    extract_total_hours,
    missing_expected_components,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LLM_EVALS") != "1",
    reason="Set RUN_LLM_EVALS=1 to run golden tests against the real LLM.",
)


@pytest.mark.eval
@pytest.mark.parametrize("golden", golden_dataset.goldens)
def test_response_schema_validity(golden) -> None:
    result = estimate_real(golden.input)

    assert isinstance(result, EstimationResponse)
    assert result.estimation.strip()
    assert result.usage.tokens_used is None or result.usage.tokens_used > 0


@pytest.mark.eval
@pytest.mark.parametrize("golden", golden_dataset.goldens)
def test_total_hours_within_expected_range(golden) -> None:
    result = estimate_real(golden.input)
    total_hours = extract_total_hours(result.estimation)
    expected_low, expected_high = golden.additional_metadata["expected_hours_range"]

    assert total_hours >= expected_low * 0.5
    assert total_hours <= expected_high * 1.5


@pytest.mark.eval
@pytest.mark.parametrize("golden", golden_dataset.goldens)
def test_expected_components_are_covered(golden) -> None:
    result = estimate_real(golden.input)
    missing_components = missing_expected_components(
        result.estimation,
        golden.additional_metadata["expected_components"],
    )

    assert missing_components == []


@pytest.mark.slow
@pytest.mark.eval
@pytest.mark.parametrize("golden", golden_dataset.goldens[:3])
def test_consistency_across_runs(golden) -> None:
    n_runs = 3
    results = [estimate_real(golden.input) for _ in range(n_runs)]
    totals = [extract_total_hours(result.estimation) for result in results]

    cv = statistics.stdev(totals) / statistics.mean(totals)
    assert cv < 0.25
