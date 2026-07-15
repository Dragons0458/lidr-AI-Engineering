"""Schemas for the Session 12 hand-written agent.

Three families of models live here:

* **Tool argument models** (``SearchBudgetsArgs`` …) — the loop validates every
  ``json.loads(function_call.arguments)`` into one of these BEFORE dispatch, so a
  malformed / hallucinated argument becomes a returned error string the model can
  self-correct from, never an exception that kills the loop.
* **Trace models** (``AgentStep`` / ``AgentTrace``) — the reasoning→action→
  observation record the exercise requires. ``AgentTrace.render`` prints the
  ``STEP N`` console format from the statement.
* **Result models** (``AgentComponent`` / ``AgentEstimate`` / ``AgentRunResult``)
  — a deliberately LIGHT final estimate, distinct from the heavy RAG ``Estimate``
  (which mandates ``SourceCitation`` / ``WorkModule`` / coherence checks). The
  terminal ``responses.parse`` call in the loop fills ``AgentEstimate``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.domain.schemas.agent_trace import AgentStep as AgentStep
from app.domain.schemas.agent_trace import AgentTrace as AgentTrace

Confidence = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Tool argument models                                                        #
# --------------------------------------------------------------------------- #
class SearchBudgetsFilters(BaseModel):
    """Optional structural filters for a budget search.

    Mirrors the nullable ``filters`` object in the tool schema. Both fields are
    optional; ``None`` means "do not constrain on this axis".
    """

    sectors: list[str] | None = Field(
        default=None,
        description="Restrict to these client sectors (e.g. ['logistics', 'industrial']).",
    )
    component_type: str | None = Field(
        default=None,
        description="Free-text hint about the kind of component (e.g. 'mobile app').",
    )


class SearchBudgetsArgs(BaseModel):
    """Validated arguments for the ``search_budgets`` tool."""

    query: str = Field(min_length=1)
    filters: SearchBudgetsFilters | None = None


class ComponentInput(BaseModel):
    """One component the agent wants costed, with its historical references."""

    name: str = Field(min_length=1)
    reference_amounts: list[float] = Field(
        description="Historical amounts (engineer-hours) for analogous work, from search_budgets."
    )


class CalculateEstimateArgs(BaseModel):
    """Validated arguments for the ``calculate_estimate`` tool."""

    components: list[ComponentInput]


class ValidateComponentInput(BaseModel):
    """One line of the estimate to validate."""

    name: str = Field(min_length=1)
    estimated_hours: float
    reference_amounts: list[float] = Field(default_factory=list)


class ValidateEstimateArgs(BaseModel):
    """Validated arguments for the ``validate_estimate`` tool."""

    components: list[ValidateComponentInput]
    total_hours: float


# --------------------------------------------------------------------------- #
# Result models                                                               #
# --------------------------------------------------------------------------- #
class AgentTaskNode(BaseModel):
    """One operational task proposed during the structure phase."""

    name: str = Field(min_length=1)
    description: str | None = None


class AgentModuleNode(BaseModel):
    """A functional module and its proposed tasks."""

    name: str = Field(min_length=1)
    description: str | None = None
    tasks: list[AgentTaskNode] = Field(default_factory=list)


class AgentStructure(BaseModel):
    """Tool-free structure proposal produced by the model."""

    modules: list[AgentModuleNode] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high", "insufficient"]
    reasoning: str
    insufficient_context_explanation: str | None = None


class AgentTaskRef(BaseModel):
    """A flagged task identified by an opaque conductor-owned reference."""

    task_ref: str = Field(min_length=1)
    module: str = Field(min_length=1)
    task: str = Field(min_length=1)
    description: str | None = None
    reason: str = Field(min_length=1)


class AgentRecoveryNeighbor(BaseModel):
    """Historical provenance used for a recovered task estimate."""

    source_id: int
    budget_id: str | None = None
    estimated_hours: int = Field(ge=0)
    distance: float = Field(ge=0)


class AgentTaskDerivation(BaseModel):
    """A deterministic consensus derivation emitted by the recovery tool."""

    task_ref: str = Field(min_length=1)
    module: str = Field(min_length=1)
    task: str = Field(min_length=1)
    estimated_hours: int | None = Field(default=None, ge=0)
    reliability: float | None = Field(default=None, ge=0, le=1)
    dispersion: float | None = Field(default=None, ge=0)
    has_match: bool
    neighbors: list[AgentRecoveryNeighbor] = Field(default_factory=list)

    @model_validator(mode="after")
    def matched_derivation_has_provenance(self) -> "AgentTaskDerivation":
        if self.has_match and (self.estimated_hours is None or not self.neighbors):
            raise ValueError(
                "has_match=true requires estimated_hours and at least one neighbor"
            )
        return self


class AgentTaskHoursRun(BaseModel):
    """Recovery-loop output; hours only originate from captured derivations."""

    derivations: list[AgentTaskDerivation] = Field(default_factory=list)
    trace: AgentTrace = Field(default_factory=AgentTrace)
    iterations: int = Field(ge=0)
    stopped_reason: Literal["completed", "max_iterations"] = "completed"


class AgentComponent(BaseModel):
    """One costed component in the final estimate."""

    name: str
    estimated_hours: float = Field(ge=0)
    cited_chunk_ids: list[int] = Field(
        default_factory=list,
        description="DB ids of the historical chunks that grounded this component.",
    )
    rationale: str = Field(description="Why this number, in one or two sentences.")


class AgentEstimate(BaseModel):
    """The agent's final structured estimate (light — no mandatory citations)."""

    components: list[AgentComponent]
    total_hours: float = Field(ge=0)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Confidence


class AgentRunResult(BaseModel):
    """Everything a single agent run produces: the estimate plus its trace."""

    estimate: AgentEstimate | None = Field(
        default=None,
        description="None when the loop stopped before producing a parseable estimate.",
    )
    trace: AgentTrace
    iterations: int = Field(ge=0, description="Number of Responses API round-trips.")
    stopped_reason: Literal["completed", "max_iterations", "no_final_estimate"] = (
        "completed"
    )
