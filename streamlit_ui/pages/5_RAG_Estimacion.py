"""RAG Estimation wizard — transcript → grounded estimate (Session 9)."""

from __future__ import annotations

import runpy
import uuid
from pathlib import Path
from typing import Any

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import streamlit as st

from streamlit_ui.common import (
    format_api_error,
    get_api_root_url,
    get_estimate_api_key,
)

st.set_page_config(page_title="RAG Estimación", page_icon="📋", layout="wide")

api_root = get_api_root_url()
estimate_key = get_estimate_api_key()
headers = {"X-API-Key": estimate_key} if estimate_key else {}

SECTORS = ["finance", "ecommerce", "healthcare", "industrial"]

_DEFAULTS = {
    "rag_transcript": "",
    "rag_reformulation": None,
    "rag_retrieval": None,
    "rag_assemble": None,
    "rag_generate": None,
    "rag_verified_modules": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _clear_downstream(*keys: str) -> None:
    for key in keys:
        st.session_state[key] = _DEFAULTS[key]


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not estimate_key:
        st.error("Configura `ESTIMATE_API_KEY` en `.env` o `.streamlit/secrets.toml`.")
        return None
    try:
        response = httpx.post(
            f"{api_root.rstrip('/')}{path}",
            json=payload,
            headers=headers,
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        st.error(format_api_error(exc, api_base_url=api_root))
        return None


st.title("RAG Estimación")
st.caption(
    "Wizard de 6 etapas: transcript → reformulación → retrieval → augmentation → "
    "generación → verificación. Equivalente al flujo `Rag::EstimationRun` de estimator-web."
)

if not estimate_key:
    st.warning(
        "Falta `ESTIMATE_API_KEY`. Los endpoints `/v1/estimate/*` requieren autenticación."
    )

with st.expander("Modo one-shot (comparación)"):
    if st.button("Ejecutar pipeline completo", key="oneshot"):
        transcript = st.session_state.rag_transcript or ""
        if len(transcript) < 100:
            st.error("El transcript debe tener al menos 100 caracteres.")
        else:
            result = _post(
                "/v1/estimate/from-transcript",
                {"transcript": transcript, "idempotency_key": str(uuid.uuid4())},
            )
            if result:
                st.json(result)

st.divider()
st.subheader("1. Transcript")
transcript = st.text_area(
    "Transcripción de la reunión",
    value=st.session_state.rag_transcript,
    height=200,
    key="transcript_input",
)
if st.button("Empezar", type="primary"):
    if len(transcript) < 100:
        st.error("Mínimo 100 caracteres.")
    else:
        st.session_state.rag_transcript = transcript
        _clear_downstream(
            "rag_reformulation",
            "rag_retrieval",
            "rag_assemble",
            "rag_generate",
            "rag_verified_modules",
        )
        result = _post("/v1/estimate/stages/reformulate", {"transcript": transcript})
        if result:
            st.session_state.rag_reformulation = result
            st.success("Reformulación completada.")

if st.session_state.rag_reformulation:
    st.divider()
    st.subheader("2. Reformulación")
    ref = st.session_state.rag_reformulation
    query = ref.get("query", {})
    col1, col2 = st.columns(2)
    with col1:
        st.json(query)
    with col2:
        st.markdown("**search_text** (para embedding)")
        st.code(ref.get("search_text", ""))
    if st.button("Re-ejecutar reformulación"):
        _clear_downstream(
            "rag_retrieval", "rag_assemble", "rag_generate", "rag_verified_modules"
        )
        result = _post(
            "/v1/estimate/stages/reformulate",
            {"transcript": st.session_state.rag_transcript},
        )
        if result:
            st.session_state.rag_reformulation = result
            st.rerun()

    st.divider()
    st.subheader("3. Retrieval")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        top_k = st.number_input("top_k", min_value=1, max_value=30, value=10)
        distance_threshold = st.number_input(
            "distance_threshold", min_value=0.0, max_value=2.0, value=0.6, step=0.05
        )
    with fcol2:
        sectors = st.multiselect("sectors", SECTORS)
        chunk_types = st.text_input("chunk_types (coma-separados)", value="")
    with fcol3:
        year_min = st.number_input(
            "project_year_min", min_value=2010, max_value=2100, value=2010
        )
        year_max = st.number_input(
            "project_year_max", min_value=2010, max_value=2100, value=2100
        )

    if st.button("Ejecutar retrieval"):
        _clear_downstream("rag_assemble", "rag_generate", "rag_verified_modules")
        payload: dict[str, Any] = {
            "query_text": ref.get("search_text", ""),
            "top_k": top_k,
            "distance_threshold": distance_threshold,
        }
        if sectors:
            payload["sectors"] = sectors
        if chunk_types.strip():
            payload["chunk_types"] = [
                c.strip() for c in chunk_types.split(",") if c.strip()
            ]
        if year_min:
            payload["project_year_min"] = int(year_min)
        if year_max:
            payload["project_year_max"] = int(year_max)
        result = _post("/v1/estimate/stages/retrieve", payload)
        if result:
            st.session_state.rag_retrieval = result
            st.rerun()

if st.session_state.rag_retrieval:
    ret = st.session_state.rag_retrieval
    st.metric("candidates_evaluated", ret.get("candidates_evaluated", 0))
    if ret.get("low_confidence"):
        st.warning("low_confidence — ningún chunk superó el umbral.")
    for chunk in ret.get("chunks", []):
        with st.expander(
            f"Chunk #{chunk['id']} · {chunk['sector']} · {chunk['project_year']} · d={chunk['distance']:.3f}"
        ):
            st.text(chunk.get("content", ""))

    st.divider()
    st.subheader("4. Augmentation")
    max_tokens = st.number_input(
        "max_context_tokens (opcional)", min_value=256, max_value=64000, value=16384
    )
    if st.button("Ensamblar contexto"):
        _clear_downstream("rag_generate", "rag_verified_modules")
        payload = {"chunks": ret.get("chunks", [])}
        if max_tokens:
            payload["max_context_tokens"] = int(max_tokens)
        result = _post("/v1/estimate/stages/assemble", payload)
        if result:
            st.session_state.rag_assemble = result
            st.rerun()

if st.session_state.rag_assemble:
    asm = st.session_state.rag_assemble
    st.caption(
        f"kept={len(asm.get('kept_chunks', []))} · dropped={asm.get('dropped_count', 0)} · tokens={asm.get('token_count', 0)}"
    )
    with st.expander("context_block"):
        st.code(asm.get("context_block", ""))

    st.divider()
    st.subheader("5. Generación")
    if st.button("Generar estimación"):
        _clear_downstream("rag_verified_modules")
        ref = st.session_state.rag_reformulation or {}
        result = _post(
            "/v1/estimate/stages/generate",
            {
                "context_block": asm.get("context_block", ""),
                "query": ref.get("query", {}),
                "kept_chunks": asm.get("kept_chunks", []),
            },
        )
        if result:
            st.session_state.rag_generate = result
            st.rerun()

if st.session_state.rag_generate:
    gen = st.session_state.rag_generate
    estimate = gen.get("estimate", {})
    st.markdown(
        f"**Confianza:** `{estimate.get('confidence')}` · **coherent:** `{gen.get('coherent')}`"
    )
    if gen.get("fabricated_source_ids"):
        st.error(f"Citas fabricadas: {gen['fabricated_source_ids']}")
    st.markdown(estimate.get("reasoning", ""))

    rows: list[dict[str, Any]] = []
    for module in estimate.get("modules", []):
        for task in module.get("tasks", []):
            rows.append(
                {
                    "module": module.get("name"),
                    "task": task.get("name"),
                    "description": task.get("description") or "",
                    "engineer_days": task.get("engineer_days", 0),
                    "sources": ",".join(str(s) for s in task.get("sources", [])),
                }
            )

    if estimate.get("sources"):
        st.markdown("**Fuentes**")
        st.dataframe(estimate["sources"], use_container_width=True, hide_index=True)
    if estimate.get("assumptions"):
        st.markdown("**Supuestos**")
        st.dataframe(estimate["assumptions"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("6. Verificación")
    if st.session_state.rag_verified_modules is None and rows:
        st.session_state.rag_verified_modules = rows

    edited = st.data_editor(
        st.session_state.rag_verified_modules or rows,
        num_rows="dynamic",
        use_container_width=True,
        key="verification_editor",
    )
    if st.button("Guardar verificación"):
        st.session_state.rag_verified_modules = edited
        total = sum(int(r.get("engineer_days") or 0) for r in edited)
        st.success(f"total_engineer_days recalculado: **{total}**")
