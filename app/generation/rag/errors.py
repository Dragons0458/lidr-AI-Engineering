"""Custom exception hierarchy for the RAG estimation pipeline (Session 9)."""

from __future__ import annotations


class RagError(Exception):
    """Base class for every error raised inside the RAG pipeline."""


class ReformulationError(RagError):
    """Query understanding failed even after the simple-rewrite fallback."""


class RetrievalError(RagError):
    """The vector store could not be queried (DB/connection failure)."""


class GenerationError(RagError):
    """The LLM generation step failed irrecoverably."""


class CitationValidationError(RagError):
    """The model kept citing fabricated source ids after the retry budget."""


class MalformedEstimateError(RagError):
    """The generated estimate violates the insufficient-context coherence rule
    even after one repair attempt."""
