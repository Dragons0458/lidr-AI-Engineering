"""RAG Chunking Lab helpers — strategy catalog and comparison rendering."""

from __future__ import annotations

from typing import Any

STRATEGY_CATALOG: list[dict[str, Any]] = [
    {
        "name": "structural",
        "label": "Structural (JSON)",
        "description": "One chunk per budget component; preserves schema boundaries.",
        "cost_tier": "free",
        "needs_key": None,
        "default_checked": True,
    },
    {
        "name": "fixed_size",
        "label": "Fixed size",
        "description": "Splits component text into fixed token windows.",
        "cost_tier": "free",
        "needs_key": None,
        "default_checked": True,
    },
    {
        "name": "recursive",
        "label": "Recursive",
        "description": "LangChain recursive splitter over serialized budgets.",
        "cost_tier": "free",
        "needs_key": None,
        "default_checked": True,
    },
    {
        "name": "sentence_window",
        "label": "Sentence window",
        "description": "One chunk per sentence with surrounding context window.",
        "cost_tier": "free",
        "needs_key": None,
        "default_checked": False,
    },
    {
        "name": "hierarchical",
        "label": "Hierarchical",
        "description": "Parent (full budget) + child (per component) chunks.",
        "cost_tier": "free",
        "needs_key": None,
        "default_checked": False,
    },
    {
        "name": "semantic",
        "label": "Semantic",
        "description": "Embedding-based breakpoints within component text.",
        "cost_tier": "cheap",
        "needs_key": "openai",
        "default_checked": False,
    },
    {
        "name": "propositional",
        "label": "Propositional",
        "description": "LLM extracts atomic propositions per component.",
        "cost_tier": "expensive",
        "needs_key": "openai",
        "default_checked": False,
    },
    {
        "name": "contextual_retrieval",
        "label": "Contextual retrieval",
        "description": "Anthropic-style contextual prefixes on each chunk.",
        "cost_tier": "expensive",
        "needs_key": "anthropic",
        "default_checked": False,
    },
]

CATALOG_BY_NAME: dict[str, dict[str, Any]] = {s["name"]: s for s in STRATEGY_CATALOG}

MODEL_KNOB_LABELS: dict[str, dict[str, str]] = {
    "PRIMARY_MODEL": {
        "label": "Primary model",
        "description": "Main LLM for estimations and generation.",
    },
    "FALLBACK_MODEL": {
        "label": "Fallback model",
        "description": "Used when the primary model is unavailable.",
    },
    "CRITIC_MODEL": {
        "label": "Critic model",
        "description": "Actor-Critic-Boss critic step; empty uses primary.",
    },
    "COMPRESSION_MODEL": {
        "label": "Compression model",
        "description": "Summarizes evicted conversation turns.",
    },
    "PROPOSITIONAL_CHUNKER_MODEL": {
        "label": "Propositional chunker model",
        "description": "LLM for propositional chunking strategy.",
    },
    "CONTEXTUAL_CHUNKER_MODEL": {
        "label": "Contextual chunker model",
        "description": "LLM for contextual retrieval chunking.",
    },
}

COST_BADGE: dict[str, str] = {
    "free": "",
    "cheap": "$",
    "expensive": "$ · slow",
}


def label_for(name: str) -> str:
    entry = CATALOG_BY_NAME.get(name)
    return entry["label"] if entry else name


def cost_hint(selected: list[str]) -> str:
    if not selected:
        return "Select at least one strategy."
    tiers = {CATALOG_BY_NAME[n]["cost_tier"] for n in selected if n in CATALOG_BY_NAME}
    if tiers == {"free"}:
        return "All selected strategies are free — expect results in seconds, zero extra LLM calls."
    if "expensive" in tiers:
        return (
            "Includes expensive strategies — may take minutes and cost ~$0.15 "
            "depending on corpus size."
        )
    if "cheap" in tiers:
        return "Includes embedding-based strategies — expect a few cents in API cost."
    return "Mixed cost tiers selected."


def build_settings_update_payload(selections: dict[str, str]) -> dict[str, str | None]:
    """Map UI selectbox values: empty string means reset to default (None)."""
    return {key: (value if value else None) for key, value in selections.items()}


def hierarchical_level_badge(chunk_id: str) -> str | None:
    if "::parent" in chunk_id:
        return "parent"
    if "::" in chunk_id:
        return "child"
    return None


def total_comparison_cost(stats_per_strategy: dict[str, Any]) -> float:
    return sum(
        float(s.get("ingestion_cost_usd", 0)) for s in stats_per_strategy.values()
    )


def render_comparison_results(response_payload: dict[str, Any], *, top_k: int) -> None:
    """Render stats table, cost bars and playground (mirrors Rails partials)."""
    import streamlit as st

    stats_per_strategy = response_payload.get("stats_per_strategy") or {}
    queries_per_strategy = response_payload.get("queries_per_strategy") or {}

    if stats_per_strategy:
        rows = []
        for name, stats in stats_per_strategy.items():
            dist = stats.get("token_distribution") or {}
            rows.append(
                {
                    "Strategy": label_for(name),
                    "Chunks": stats.get("n_chunks"),
                    "Min tok": dist.get("min"),
                    "P50": dist.get("p50"),
                    "P95": dist.get("p95"),
                    "Max tok": dist.get("max"),
                    "Orphans (<20)": stats.get("n_orphan_chunks"),
                    "Obese (>800)": stats.get("n_obese_chunks"),
                    "Cost ($)": stats.get("ingestion_cost_usd"),
                    "Seconds": stats.get("ingestion_seconds"),
                }
            )
        st.subheader("Corpus stats")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        costs = [
            float(s.get("ingestion_cost_usd", 0)) for s in stats_per_strategy.values()
        ]
        max_cost = max(costs) if costs else 0.0
        st.subheader("Relative cost")
        if max_cost == 0:
            st.info("Zero extra LLM calls for the selected strategies.")
        else:
            for name, stats in stats_per_strategy.items():
                cost = float(stats.get("ingestion_cost_usd", 0))
                st.caption(f"{label_for(name)} — ${cost:.4f}")
                st.progress(cost / max_cost if max_cost else 0.0)

    if queries_per_strategy:
        st.subheader("Playground")
        all_queries: list[str] = []
        for results in queries_per_strategy.values():
            for qr in results:
                q = qr.get("query", "")
                if q and q not in all_queries:
                    all_queries.append(q)

        for idx, query in enumerate(all_queries):
            with st.expander(query, expanded=idx == 0):
                for strategy_name in stats_per_strategy:
                    strategy_results = queries_per_strategy.get(strategy_name, [])
                    match = next(
                        (r for r in strategy_results if r.get("query") == query), None
                    )
                    if not match:
                        continue
                    st.markdown(f"**{label_for(strategy_name)}**")
                    for chunk in match.get("top_k", [])[:top_k]:
                        cosine = float(chunk.get("cosine", 0))
                        chunk_id = str(chunk.get("chunk_id", ""))
                        preview = chunk.get("text_preview", "")
                        level = hierarchical_level_badge(chunk_id)
                        badge = (
                            f" · `{level}`"
                            if level and strategy_name == "hierarchical"
                            else ""
                        )
                        st.caption(f"`{chunk_id}`{badge} — cosine {cosine:.3f}")
                        st.progress(min(max(cosine, 0.0), 1.0))
                        st.text(preview)
