from __future__ import annotations

import structlog
import tiktoken

from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk

log = structlog.get_logger()


class JSONStructuralChunker:
    """Create one embedding chunk per budget component with parent context."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for budget in budgets:
            for component in budget.components:
                text = self._build_text(budget, component)
                token_count = len(self._encoding.encode(text))
                if token_count > 8000:
                    log.warning(
                        "embedding_chunk_large",
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        token_count=token_count,
                    )
                chunks.append(
                    Chunk(
                        chunk_id=f"{budget.budget_id}::{component.component_id}",
                        text=text,
                        metadata={
                            "budget_id": budget.budget_id,
                            "component_id": component.component_id,
                            "client_sector": budget.client_metadata.sector,
                            "main_technology": budget.main_technology,
                            "year": budget.year,
                            "complexity": component.complexity,
                            "estimated_hours": component.estimated_hours,
                        },
                        token_count=token_count,
                    )
                )
        return chunks

    @staticmethod
    def _build_text(budget: Budget, component: BudgetComponent) -> str:
        dependencies = ", ".join(component.dependencies) or "None"
        tech_stack = ", ".join(component.tech_stack)
        return "\n".join(
            [
                f"Project: {budget.project_summary}",
                (
                    "Client sector | Year | Main tech: "
                    f"{budget.client_metadata.sector} | {budget.year} | "
                    f"{budget.main_technology}"
                ),
                f"Component: {component.name}",
                f"Description: {component.description}",
                f"Tech stack: {tech_stack}",
                f"Complexity: {component.complexity}",
                f"Estimated hours: {component.estimated_hours}",
                f"Dependencies: {dependencies}",
            ]
        )
