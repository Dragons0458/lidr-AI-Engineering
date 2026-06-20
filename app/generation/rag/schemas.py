from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class BudgetComponent(BaseModel):
    component_id: str
    name: str
    description: str
    tech_stack: list[str]
    estimated_hours: int = Field(ge=0)
    complexity: Complexity
    dependencies: list[str] = Field(default_factory=list)


class ClientMetadata(BaseModel):
    name: str
    sector: Sector
    country: str


class Budget(BaseModel):
    budget_id: str
    client_metadata: ClientMetadata
    project_summary: str
    main_technology: str
    year: int
    total_estimated_hours: int = Field(ge=0)
    components: list[BudgetComponent]


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    token_count: int = Field(ge=0)


class EmbeddedChunk(Chunk):
    embedding: list[float]


class IngestRequest(BaseModel):
    source_path: str
    document_type: str
    content: Budget


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int
    embedding_dimension: int
    ingestion_time_ms: int


class SearchRequest(BaseModel):
    query: str
    k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    k: int
    search_time_ms: int
    results: list[SearchHit]


# ---------------------------------------------------------------------------
# Session 9 — RAG estimation pipeline (query understanding → generation).
# ---------------------------------------------------------------------------

Scale = Literal["small", "medium", "large", "unknown"]
Confidence = Literal["high", "medium", "low", "insufficient"]
Relevance = Literal["primary", "supporting", "tangential"]
Impact = Literal["high", "medium", "low"]


class EstimationQuery(BaseModel):
    """Structured brief distilled from a raw meeting transcript."""

    function: str = Field(description="Functional summary of the project.")
    technologies: list[str] = Field(default_factory=list)
    sector: str | None = None
    scale: Scale = "unknown"
    country: str | None = None
    regulations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    id: int
    content: str
    sector: str
    project_year: int
    chunk_type: str
    distance: float = Field(description="Cosine distance (lower = more similar).")


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    low_confidence: bool = Field(
        description="True when no chunk crossed the distance threshold (soft-fail)."
    )
    candidates_evaluated: int = Field(
        ge=0, description="Total chunks scored before applying the threshold/limit."
    )


class SourceCitation(BaseModel):
    source_id: int = Field(
        description="DB id of the cited chunk (a RetrievedChunk.id)."
    )
    relevance: Relevance
    used_for: str = Field(description="What this source contributed to the estimate.")


class Assumption(BaseModel):
    description: str
    impact: Impact
    rationale: str


class TaskItem(BaseModel):
    name: str
    description: str | None = Field(
        default=None, description="One-line scope of the task."
    )
    engineer_days: int = Field(ge=0)
    sources: list[int] = Field(
        default_factory=list, description="Chunk ids that back this task."
    )


class WorkModule(BaseModel):
    name: str
    description: str | None = Field(
        default=None, description="What this functional block covers."
    )
    tasks: list[TaskItem] = Field(default_factory=list)


class Estimate(BaseModel):
    total_engineer_days: int | None = None
    modules: list[WorkModule] = Field(default_factory=list)
    duration_weeks: int | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = Field(description="How the estimate was derived from the sources.")
    insufficient_context_explanation: str | None = None


class RetrievalRequest(BaseModel):
    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0.0, le=2.0)
    sectors: list[str] | None = None
    project_year_min: int | None = Field(default=None, ge=2010, le=2100)
    project_year_max: int | None = Field(default=None, ge=2010, le=2100)
    chunk_types: list[str] | None = None


class EstimateRequest(BaseModel):
    transcript: str = Field(min_length=100, max_length=50_000)
    idempotency_key: str | None = Field(default=None, max_length=128)


class ReformulateRequest(BaseModel):
    transcript: str = Field(min_length=100, max_length=50_000)


class ReformulationResult(BaseModel):
    query: EstimationQuery
    search_text: str = Field(description="Corpus-aligned text fed to the embedder.")


class AssembleRequest(BaseModel):
    chunks: list[RetrievedChunk]
    max_context_tokens: int | None = Field(default=None, ge=256, le=64_000)


class AssembleResult(BaseModel):
    context_block: str
    kept_chunks: list[RetrievedChunk]
    dropped_count: int = Field(ge=0, description="Chunks dropped by the token budget.")
    token_count: int = Field(ge=0, description="Tokens in the assembled context block.")


class GenerateRequest(BaseModel):
    context_block: str = Field(min_length=1)
    query: EstimationQuery
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)


class GenerateResult(BaseModel):
    estimate: Estimate
    fabricated_source_ids: list[int] = Field(
        default_factory=list,
        description="Cited source ids not present in kept_chunks (empty = clean).",
    )
    coherent: bool = Field(
        description="False when an insufficient estimate still carries numbers."
    )
