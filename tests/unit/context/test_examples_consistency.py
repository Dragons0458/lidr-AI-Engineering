from app.context.examples import (
    ESTIMATION_EXAMPLES,
    format_example_json,
    format_example_markdown,
    format_example_narrative,
)


def test_canonical_examples_have_consistent_total_hours() -> None:
    for example in ESTIMATION_EXAMPLES:
        assert sum(phase.hours for phase in example.phases) == example.total_hours


def test_example_formatters_return_non_empty_text() -> None:
    for example in ESTIMATION_EXAMPLES:
        assert format_example_markdown(example).strip()
        assert format_example_json(example).strip()
        assert format_example_narrative(example).strip()
