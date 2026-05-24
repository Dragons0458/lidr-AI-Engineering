import pytest

from evals.stress.scenarios import (
    SCENARIO_LENGTHS,
    SCENARIOS,
    get_scenario,
    iter_scenarios,
)


def test_each_stress_scenario_has_twenty_ordered_turns() -> None:
    assert sorted(SCENARIOS) == ["contradiction", "growing", "pivot"]

    for scenario in SCENARIOS.values():
        assert len(scenario.turns) == 20
        assert [turn.turn_index for turn in scenario.turns] == list(range(1, 21))
        assert all(turn.fact_to_remember for turn in scenario.turns)
        assert all(
            turn.fact_to_remember.lower() in turn.transcript.lower()
            for turn in scenario.turns
        )


def test_scenario_truncate_keeps_first_n_turns() -> None:
    scenario = get_scenario("growing", max_turns=3)

    assert scenario.name == "growing_3"
    assert [turn.turn_index for turn in scenario.turns] == [1, 2, 3]
    assert scenario.turns[0].fact_to_remember == "project name: Nimbus"


def test_facts_before_returns_only_previous_facts() -> None:
    scenario = get_scenario("contradiction")

    assert scenario.facts_before(1) == ()
    assert scenario.facts_before(4) == (
        "project name: LedgerFlow",
        "scope includes invoice approvals",
        "budget locked: 30000 EUR",
    )


def test_iter_scenarios_expands_names_and_lengths() -> None:
    scenarios = list(iter_scenarios(["pivot"], lengths=(1, 6)))

    assert SCENARIO_LENGTHS == (1, 3, 6, 10, 20)
    assert [scenario.name for scenario in scenarios] == ["pivot_1", "pivot_6"]
    assert [len(scenario.turns) for scenario in scenarios] == [1, 6]


def test_get_scenario_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("missing")
