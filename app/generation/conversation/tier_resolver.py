"""Adaptive audience tier resolution for structured estimation prompts."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import structlog

from app.generation.conversation.store import ProjectMetadata

log = structlog.get_logger()

_NDA_PATTERN = re.compile(
    r"\b(nda|non[- ]disclosure|confidential|legal\s+hold)\b",
    re.IGNORECASE,
)
_REGULATORY_PATTERN = re.compile(
    r"\b("
    r"hipaa|gdpr|sox|pci[- ]dss|fda|iso[- ]?27001|ccpa"
    r")\b",
    re.IGNORECASE,
)
_TECH_KEYWORDS = (
    "docker",
    "kubernetes",
    "microservice",
    "terraform",
    "grpc",
    "graphql",
    "kafka",
    "airflow",
    "spark",
    "rabbitmq",
)
_TECH_PATTERN = re.compile(
    r"\b(" + "|".join(_TECH_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class Tier(str, Enum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass(frozen=True)
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata
    override: Tier | None = None


@dataclass(frozen=True)
class TierRule:
    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


def _combined_text(ctx: ResolutionContext) -> str:
    parts = [ctx.transcript]
    if ctx.metadata.agreed_scope:
        parts.append(ctx.metadata.agreed_scope)
    if ctx.metadata.mentioned_technologies:
        parts.extend(ctx.metadata.mentioned_technologies)
    return "\n".join(parts)


def _has_nda(ctx: ResolutionContext) -> bool:
    return bool(_NDA_PATTERN.search(_combined_text(ctx)))


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    return bool(_REGULATORY_PATTERN.search(_combined_text(ctx)))


def _technical_audience(ctx: ResolutionContext) -> bool:
    matches = {m.group(1).lower() for m in _TECH_PATTERN.finditer(ctx.transcript)}
    return len(matches) >= 2


def _is_small_team(ctx: ResolutionContext) -> bool:
    size = ctx.metadata.assumed_team_size
    return size is not None and size <= 2


_TIER_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    override: Tier | None = None,
) -> tuple[Tier, str]:
    """Resolve audience tier; override wins, else first matching rule, else default."""
    if override is not None:
        log.info("tier_resolved", tier=override.value, rule="override")
        return override, "override"

    ctx = ResolutionContext(transcript=transcript, metadata=metadata)
    for rule in _TIER_RULES:
        try:
            if rule.predicate(ctx):
                log.info("tier_resolved", tier=rule.tier.value, rule=rule.name)
                return rule.tier, rule.name
        except Exception as exc:
            log.warning(
                "tier_rule_predicate_failed",
                rule=rule.name,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    log.info("tier_resolved", tier=Tier.DEFAULT.value, rule="default")
    return Tier.DEFAULT, "default"
