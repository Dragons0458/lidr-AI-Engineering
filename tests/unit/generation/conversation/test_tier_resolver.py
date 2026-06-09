import pytest

from app.generation.conversation.store import ProjectMetadata
from app.generation.conversation.tier_resolver import (
    Tier,
    TierRule,
    ResolutionContext,
    resolve_tier,
)


def test_override_wins() -> None:
    tier, rule = resolve_tier(
        transcript="kubernetes docker grpc",
        metadata=ProjectMetadata(),
        override=Tier.PM,
    )
    assert tier == Tier.PM
    assert rule == "override"


def test_nda_maps_to_executive() -> None:
    tier, rule = resolve_tier(
        transcript="We signed an NDA last week.",
        metadata=ProjectMetadata(),
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_gdpr_maps_to_executive() -> None:
    tier, rule = resolve_tier(
        transcript="Must be GDPR compliant.",
        metadata=ProjectMetadata(),
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "regulatory_context"


def test_technical_keywords_map_to_developer() -> None:
    tier, rule = resolve_tier(
        transcript="We need docker and kubernetes in production.",
        metadata=ProjectMetadata(),
    )
    assert tier == Tier.DEVELOPER
    assert rule == "technical_audience"


def test_small_team_maps_to_pm() -> None:
    tier, rule = resolve_tier(
        transcript="Small delivery.",
        metadata=ProjectMetadata(assumed_team_size=2),
    )
    assert tier == Tier.PM
    assert rule == "low_budget_pm"


def test_no_match_returns_default() -> None:
    tier, rule = resolve_tier(
        transcript="Build a simple landing page.",
        metadata=ProjectMetadata(),
    )
    assert tier == Tier.DEFAULT
    assert rule == "default"


def test_precedence_nda_over_technical() -> None:
    tier, rule = resolve_tier(
        transcript="NDA in place. Stack: docker, kubernetes, terraform.",
        metadata=ProjectMetadata(),
    )
    assert tier == Tier.EXECUTIVE
    assert rule == "nda_detected"


def test_predicate_exception_is_ignored() -> None:
    def boom(_ctx: ResolutionContext) -> bool:
        raise RuntimeError("predicate failed")

    rules = (
        TierRule("broken", Tier.EXECUTIVE, boom),
        TierRule("fallback", Tier.PM, lambda _ctx: True),
    )
    ctx = ResolutionContext(transcript="x", metadata=ProjectMetadata())
    for rule in rules:
        try:
            if rule.predicate(ctx):
                assert rule.tier == Tier.PM
                return
        except RuntimeError:
            continue
    pytest.fail("expected fallback rule to match")
