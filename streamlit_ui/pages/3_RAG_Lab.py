"""RAG Chunking Lab — compare strategies and browse local run history."""

from __future__ import annotations

import json
import runpy
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import streamlit as st

from streamlit_ui.common import format_api_error, get_api_base_url, get_api_root_url
from streamlit_ui.rag import (
    COST_BADGE,
    STRATEGY_CATALOG,
    cost_hint,
    label_for,
    render_comparison_results,
    total_comparison_cost,
)
from streamlit_ui.store import get_comparison, list_comparisons, save_comparison

st.set_page_config(page_title="RAG Lab", page_icon="🧪", layout="wide")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CORPUS_PATH = DATA_DIR / "budgets_sample.json"
QUERIES_PATH = DATA_DIR / "test_queries.json"

api_base_url = get_api_base_url()
api_root = get_api_root_url(api_base_url)

if "rag_queries" not in st.session_state:
    st.session_state.rag_queries: list[str] = []

st.title("RAG Chunking Lab")
st.caption(f"`POST {api_root.rstrip('/')}/embeddings/compare`")

st.subheader("Estrategias")
selected: list[str] = []
for entry in STRATEGY_CATALOG:
    badge = COST_BADGE.get(entry["cost_tier"], "")
    needs = f" · needs {entry['needs_key']}" if entry.get("needs_key") else ""
    checked = st.checkbox(
        f"{entry['label']}{(' · ' + badge) if badge else ''}{needs}",
        value=entry["default_checked"],
        key=f"strategy_{entry['name']}",
    )
    st.caption(entry["description"])
    if checked:
        selected.append(entry["name"])

st.info(cost_hint(selected))

st.subheader("Queries (playground)")
if QUERIES_PATH.exists():
    benchmark_queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    chip_cols = st.columns(min(len(benchmark_queries), 3))
    for idx, query in enumerate(benchmark_queries):
        if chip_cols[idx % len(chip_cols)].button(query[:40], key=f"chip_{idx}"):
            if query not in st.session_state.rag_queries:
                st.session_state.rag_queries.append(query)

for idx, query in enumerate(list(st.session_state.rag_queries)):
    cols = st.columns([5, 1])
    cols[0].text_input("Query", value=query, key=f"rag_q_{idx}", disabled=True)
    if cols[1].button("✕", key=f"rm_q_{idx}"):
        st.session_state.rag_queries.pop(idx)
        st.rerun()

new_query = st.text_input("Añadir query libre")
if st.button("Añadir query") and new_query.strip():
    st.session_state.rag_queries.append(new_query.strip())
    st.rerun()

top_k = st.slider("top_k", min_value=1, max_value=10, value=3)

if st.button("Comparar estrategias", type="primary", disabled=not selected):
    if not CORPUS_PATH.exists():
        st.error(f"Corpus no encontrado: {CORPUS_PATH}")
    else:
        budgets = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        payload = {
            "budgets": budgets,
            "strategies": selected,
            "queries": st.session_state.rag_queries,
            "top_k": top_k,
        }
        endpoint = f"{api_root.rstrip('/')}/embeddings/compare"
        with st.spinner("Comparando estrategias (puede tardar varios minutos)…"):
            start = time.perf_counter()
            try:
                response = httpx.post(endpoint, json=payload, timeout=600.0)
                response.raise_for_status()
                body = response.json()
                duration_ms = int((time.perf_counter() - start) * 1000)
                comparison_id = save_comparison(
                    strategies=selected,
                    queries=st.session_state.rag_queries,
                    top_k=top_k,
                    corpus_label="budgets_sample",
                    corpus_count=len(budgets),
                    response_payload=body,
                    duration_ms=duration_ms,
                )
                st.session_state.last_comparison_id = comparison_id
                st.session_state.last_comparison_body = body
                st.success(
                    f"Comparación guardada (#{comparison_id}) en {duration_ms} ms."
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (500, 503):
                    st.error(
                        format_api_error(exc, api_base_url=api_base_url)
                        + "\n\n_Comprueba que OPENAI_API_KEY (y ANTHROPIC_API_KEY si aplica) "
                        "estén configuradas._"
                    )
                else:
                    st.error(format_api_error(exc, api_base_url=api_base_url))
            except httpx.HTTPError as exc:
                st.error(format_api_error(exc, api_base_url=api_base_url))

if st.session_state.get("last_comparison_body"):
    render_comparison_results(
        st.session_state.last_comparison_body,
        top_k=top_k,
    )

st.divider()
st.subheader("Histórico de comparaciones")
runs = list_comparisons(limit=20)
if not runs:
    st.caption("Sin comparaciones guardadas.")
else:
    for run in runs:
        payload = run.get("response_payload") or {}
        strategies = run.get("strategies") or []
        queries = run.get("queries") or []
        total_cost = total_comparison_cost(payload.get("stats_per_strategy") or {})
        cols = st.columns([4, 1, 1, 1])
        strategy_labels = ", ".join(label_for(s) for s in strategies[:3])
        if len(strategies) > 3:
            strategy_labels += f" +{len(strategies) - 3}"
        cols[0].write(
            f"**#{run['id']}** · {strategy_labels} · "
            f"{len(queries)} queries · top_k={run['top_k']}"
        )
        cols[1].caption(f"${total_cost:.4f}")
        cols[2].caption(f"{run.get('duration_ms', 0)} ms")
        if cols[3].button("Ver", key=f"view_cmp_{run['id']}"):
            st.session_state.view_comparison_id = run["id"]
            st.rerun()

if st.session_state.get("view_comparison_id"):
    record = get_comparison(st.session_state.view_comparison_id)
    if record:
        with st.expander(f"Comparación #{record['id']}", expanded=True):
            render_comparison_results(
                record.get("response_payload") or {},
                top_k=int(record.get("top_k") or 3),
            )
