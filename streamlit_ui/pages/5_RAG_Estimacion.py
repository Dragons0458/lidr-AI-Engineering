"""Session 12 hybrid RAG estimation wizard."""

from __future__ import annotations

import copy
import json
import runpy
import uuid
from pathlib import Path
from typing import Any

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import httpx
import streamlit as st

from streamlit_ui.agent_estimation import (
    ONE_SHOT_PATH,
    REFORMULATE_PATH,
    calculate_totals,
    estimate_to_rows,
    mark_manual_edits,
    normalize_trace,
    post_agent_hours,
    post_agent_structure,
    post_deterministic_hours,
    post_deterministic_structure,
    post_json,
    rows_to_modules,
    task_hours_to_rows,
    trace_counts,
)
from streamlit_ui.agents import (
    STREAMLIT_DEFAULT_HOURLY_RATE_EUR,
    avatar_data_uri,
    phase_payload,
    profile_summary,
)
from streamlit_ui.common import (
    fetch_available_agent_models,
    format_api_error,
    get_api_root_url,
    get_estimate_api_key,
)
from streamlit_ui.rag import verify_estimate
from streamlit_ui.store import (
    clear_rag_run_downstream,
    confirm_rag_estimation_run,
    create_rag_estimation_run,
    get_default_agent_profile,
    get_rag_estimation_run,
    list_agent_profiles,
    update_rag_estimation_run,
)

st.set_page_config(page_title="RAG Estimación S12", page_icon="📋", layout="wide")

api_root = get_api_root_url()
estimate_key = get_estimate_api_key()
profiles = list_agent_profiles()
profile_by_id = {profile["id"]: profile for profile in profiles}
default_profile = get_default_agent_profile()
available_models = fetch_available_agent_models(f"{api_root}/api/v1")

_DEFAULTS = {
    "rag_run_id": None,
    "rag_mode": "agentic",
    "rag_transcript": "",
    "rag_reformulation": None,
    "rag_structure": None,
    "rag_structure_rows": None,
    "rag_hours": None,
    "rag_gate_report": None,
    "rag_final_rows": None,
    "rag_structure_profile_id": default_profile["id"] if default_profile else None,
    "rag_hours_profile_id": default_profile["id"] if default_profile else None,
}
for key, default in _DEFAULTS.items():
    st.session_state.setdefault(key, default)


def _records(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return [dict(row) for row in value]


def _profile(profile_id: int | None) -> dict[str, Any] | None:
    return profile_by_id.get(profile_id)


def _profile_snapshot(profile_id: int | None) -> dict[str, Any] | None:
    profile = _profile(profile_id)
    if profile is None:
        return None
    snapshot = dict(profile)
    snapshot.pop("avatar_bytes", None)
    snapshot.pop("config", None)
    return snapshot


def _model_available(profile: dict[str, Any] | None) -> bool:
    model = ((profile or {}).get("config_payload") or {}).get("model")
    return not model or model in available_models


def _post(call: Any, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
    if not estimate_key:
        st.error("Configura `ESTIMATE_API_KEY` en `.env` o secrets.")
        return None
    try:
        return call(*args, api_key=estimate_key, **kwargs)
    except httpx.HTTPError as exc:
        st.error(format_api_error(exc, api_base_url=api_root))
        return None
    except (TypeError, ValueError) as exc:
        st.error(str(exc))
        return None


def _save_error(stage: str, message: str) -> None:
    run_id = st.session_state.rag_run_id
    if run_id:
        update_rag_estimation_run(
            run_id, status="failed", current_step=stage, last_error=message
        )


def _render_profile(profile: dict[str, Any] | None) -> None:
    if profile is None:
        st.info("Configuración del servicio")
        return
    cols = st.columns([1, 5])
    with cols[0]:
        avatar = avatar_data_uri(profile)
        if avatar:
            st.image(avatar, width=72)
        else:
            st.markdown("## 🤖")
    with cols[1]:
        st.markdown(f"**{profile['name']}**")
        st.caption(profile_summary(profile))
        if profile["persona"]:
            st.caption(profile["persona"])
        if not _model_available(profile):
            st.error("El modelo guardado no está disponible. Sustituye el modelo.")


def _render_trace(title: str, raw_trace: dict[str, Any] | None) -> None:
    trace = normalize_trace(raw_trace)
    with st.expander(title):
        if not trace["steps"]:
            st.caption("Sin pasos de agente.")
        for step in trace["steps"]:
            st.markdown(f"**Paso {step['step']} · {step['tool']}**")
            st.write(step["reasoning_summary"])
            st.json(step["tool_args"])
            st.code(step["observation"])


def _profile_picker(label: str, state_key: str, run_id: int) -> int | None:
    options = [None, *profile_by_id]
    current = st.session_state.get(state_key)
    if current not in options:
        current = default_profile["id"] if default_profile else None
    selected = st.selectbox(
        label,
        options,
        index=options.index(current),
        format_func=lambda value: (
            "Configuración del servicio"
            if value is None
            else profile_by_id[value]["name"]
        ),
        key=f"{state_key}_{run_id}",
    )
    st.session_state[state_key] = selected
    _render_profile(_profile(selected))
    return selected


st.title("RAG Estimación · Sesión 12")
st.caption(
    "Flujo híbrido: transcript → estructura agéntica → revisión humana → "
    "horas deterministas + recovery selectivo → confirmación."
)

mode_label = st.radio(
    "Modo",
    ["Agéntico", "Determinista"],
    index=0 if st.session_state.rag_mode == "agentic" else 1,
    horizontal=True,
    key="rag_mode_selector",
)
selected_mode = "agentic" if mode_label == "Agéntico" else "deterministic"

with st.expander("Modo one-shot (comparación)"):
    if st.button("Ejecutar pipeline completo", key="oneshot"):
        transcript = st.session_state.rag_transcript or ""
        if len(transcript) < 100:
            st.error("El transcript debe tener al menos 100 caracteres.")
        else:
            result = _post(
                post_json,
                api_root,
                ONE_SHOT_PATH,
                {"transcript": transcript, "idempotency_key": str(uuid.uuid4())},
            )
            if result:
                if st.session_state.rag_run_id:
                    update_rag_estimation_run(
                        st.session_state.rag_run_id, one_shot_result=result
                    )
                st.json(result)

st.divider()
st.subheader("1. Transcript")
transcript = st.text_area(
    "Transcripción de la reunión",
    value=st.session_state.rag_transcript,
    height=200,
    key=f"rag_transcript_input_{st.session_state.rag_run_id or 'new'}",
)
if st.button("Empezar", type="primary"):
    if len(transcript) < 100:
        st.error("El transcript debe tener al menos 100 caracteres.")
    else:
        run_id = create_rag_estimation_run(mode=selected_mode, transcript=transcript)
        st.session_state.update(
            {
                **_DEFAULTS,
                "rag_run_id": run_id,
                "rag_mode": selected_mode,
                "rag_transcript": transcript,
            }
        )
        result = _post(
            post_json,
            api_root,
            REFORMULATE_PATH,
            {"transcript": transcript},
        )
        if result:
            st.session_state.rag_reformulation = result
            update_rag_estimation_run(
                run_id,
                reformulation_payload=result,
                current_step="reformulation",
                status="draft",
            )
            st.rerun()
        _save_error("reformulation", "Reformulation request failed.")

run_id = st.session_state.rag_run_id
if run_id:
    persisted_run = get_rag_estimation_run(run_id)
    if persisted_run and persisted_run["status"] == "confirmed":
        st.success("Este run está confirmado y es inmutable.")
        st.json(persisted_run.get("final_rows") or [])
        st.stop()

if st.session_state.rag_reformulation and run_id:
    ref = st.session_state.rag_reformulation
    query = ref.get("query") or {}
    st.divider()
    st.subheader("2. Reformulación")
    c1, c2 = st.columns(2)
    c1.json(query)
    c2.code(ref.get("search_text", ""))

    st.divider()
    st.subheader("3. Propuesta de estructura")
    structure_profile_id = _profile_picker(
        "Perfil de estructura",
        "rag_structure_profile_id",
        run_id,
    )
    structure_profile = _profile(structure_profile_id)
    if st.button("Generar estructura", key=f"generate_structure_{run_id}"):
        if st.session_state.rag_mode == "agentic" and not _model_available(
            structure_profile
        ):
            st.error("Selecciona un perfil con un modelo disponible.")
        else:
            clear_rag_run_downstream(run_id, "structure")
            if st.session_state.rag_mode == "agentic":
                result = _post(
                    post_agent_structure,
                    api_root,
                    query,
                    phase_payload(structure_profile, "structure"),
                )
            else:
                result = _post(post_deterministic_structure, api_root, query)
            if result:
                rows = estimate_to_rows(result.get("estimate"))
                st.session_state.rag_structure = result
                st.session_state.rag_structure_rows = rows
                st.session_state.rag_hours = None
                st.session_state.rag_gate_report = None
                st.session_state.rag_final_rows = None
                update_rag_estimation_run(
                    run_id,
                    mode=st.session_state.rag_mode,
                    structure_response=result,
                    reviewed_structure=rows,
                    structure_profile_id=structure_profile_id,
                    structure_profile_snapshot=_profile_snapshot(structure_profile_id),
                    status="structure_review",
                    current_step="structure",
                )
                st.rerun()
            _save_error("structure", "Structure request failed.")

if st.session_state.rag_structure and run_id:
    _render_trace(
        "Traza de estructura",
        st.session_state.rag_structure.get("agent_trace"),
    )

if st.session_state.rag_structure_rows is not None and run_id:
    st.caption("Edita, añade o elimina módulos y tareas antes de estimar horas.")
    before = copy.deepcopy(st.session_state.rag_structure_rows)
    edited = _records(
        st.data_editor(
            before,
            num_rows="dynamic",
            use_container_width=True,
            key=f"structure_editor_{run_id}",
        )
    )
    if edited != before:
        clear_rag_run_downstream(run_id, "structure_review")
        st.session_state.rag_structure_rows = edited
        st.session_state.rag_hours = None
        st.session_state.rag_gate_report = None
        st.session_state.rag_final_rows = None
        update_rag_estimation_run(
            run_id,
            reviewed_structure=edited,
            status="structure_review",
            current_step="structure_review",
        )

    st.divider()
    st.subheader("4. Horas por tarea")
    hours_profile_id = _profile_picker(
        "Perfil de recovery",
        "rag_hours_profile_id",
        run_id,
    )
    hours_profile = _profile(hours_profile_id)
    if st.button("Estimar horas", key=f"estimate_hours_{run_id}"):
        modules = rows_to_modules(edited)
        if not modules:
            st.error("Añade al menos un módulo con tareas.")
        elif st.session_state.rag_mode == "agentic" and not _model_available(
            hours_profile
        ):
            st.error("Selecciona un perfil con un modelo disponible.")
        else:
            clear_rag_run_downstream(run_id, "hours")
            if st.session_state.rag_mode == "agentic":
                result = _post(
                    post_agent_hours,
                    api_root,
                    modules,
                    phase_payload(hours_profile, "hours"),
                )
            else:
                result = _post(post_deterministic_hours, api_root, modules)
            if result:
                final_rows = task_hours_to_rows(
                    result, hourly_rate_eur=STREAMLIT_DEFAULT_HOURLY_RATE_EUR
                )
                st.session_state.rag_hours = result
                st.session_state.rag_final_rows = final_rows
                update_rag_estimation_run(
                    run_id,
                    reviewed_structure=edited,
                    task_hours_response=result,
                    final_rows=final_rows,
                    hours_profile_id=hours_profile_id,
                    hours_profile_snapshot=_profile_snapshot(hours_profile_id),
                    status="hours_review",
                    current_step="hours",
                )
                st.rerun()
            _save_error("hours", "Task-hours request failed.")

if st.session_state.rag_hours and run_id:
    result = st.session_state.rag_hours
    tasks = result.get("tasks") or []
    trace = result.get("agent_trace")
    counts = trace_counts(trace)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pasos", counts["steps"])
    c2.metric("Búsquedas", counts["search_budgets"])
    c3.metric("Derivaciones", counts["derive_task_hours"])
    _render_trace("Traza de recovery", trace)

    unresolved = [
        task
        for task in tasks
        if not task.get("has_match") or task.get("estimated_hours") is None
    ]
    if (
        st.session_state.rag_mode == "agentic"
        and not counts["steps"]
        and not unresolved
    ):
        st.success("Recovery no necesario: todas las tareas quedaron resueltas.")
    for task in unresolved:
        st.warning(f"Sin resolver: **{task.get('module')} → {task.get('task')}**")
    for task in tasks:
        details = []
        if task.get("reliability") is not None:
            details.append(f"reliability {task['reliability']:.2f}")
        if task.get("dispersion") is not None:
            details.append(f"dispersion {task['dispersion']:.2f}")
        details.append(f"source {task.get('estimation_source', 'deterministic')}")
        st.caption(
            f"**{task.get('module')} → {task.get('task')}** · " + " · ".join(details)
        )
        if task.get("hours_range"):
            st.info(json.dumps(task["hours_range"], ensure_ascii=False))
        if task.get("neighbors"):
            with st.expander(f"Vecinos · {task.get('task')}"):
                st.json(task["neighbors"])

    st.divider()
    st.subheader("5. Hallucination gate (opcional)")
    if st.button("Verificar líneas", key=f"gate_{run_id}"):
        estimate_stub = {
            "confidence": "high",
            "reasoning": "human-reviewed hybrid wizard",
            "modules": [
                {
                    "name": module["name"],
                    "tasks": [
                        {
                            **task,
                            "engineer_days": None,
                            "grounded": False,
                            "sources": [],
                        }
                        for task in module["tasks"]
                    ],
                }
                for module in rows_to_modules(st.session_state.rag_structure_rows or [])
            ],
        }
        try:
            report = verify_estimate(
                api_root,
                estimate=estimate_stub,
                kept_chunks=[],
                estimate_key=estimate_key,
                use_judge=False,
            )
            st.session_state.rag_gate_report = report
            update_rag_estimation_run(run_id, gate_report=report, current_step="gate")
        except Exception as exc:  # noqa: BLE001
            _save_error("gate", str(exc))
            st.error(format_api_error(exc, api_base_url=api_root))
    if st.session_state.rag_gate_report:
        st.json(st.session_state.rag_gate_report)

    st.divider()
    st.subheader("6. Revisión final")
    original_rows = task_hours_to_rows(
        result, hourly_rate_eur=STREAMLIT_DEFAULT_HOURLY_RATE_EUR
    )
    edited_final = _records(
        st.data_editor(
            st.session_state.rag_final_rows,
            num_rows="dynamic",
            use_container_width=True,
            disabled=[
                "module",
                "task",
                "source",
                "reliability",
                "dispersion",
                "has_match",
                "neighbors",
                "hours_range",
                "cost_eur",
            ],
            key=f"final_hours_editor_{run_id}",
        )
    )
    final_rows = mark_manual_edits(original_rows, edited_final)
    st.session_state.rag_final_rows = final_rows
    totals = calculate_totals(final_rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total horas", totals["total_hours"])
    c2.metric("Engineer-days (8h)", totals["total_engineer_days"])
    c3.metric("Coste EUR", f"{totals['total_cost_eur']:.2f} €")
    st.caption("Cada coste se calcula como horas × tarifa EUR/h.")
    if st.button(
        "Confirmar estimación",
        type="primary",
        key=f"confirm_{run_id}",
    ):
        confirm_rag_estimation_run(
            run_id,
            final_rows=final_rows,
            **totals,
            structure_profile_snapshot=_profile_snapshot(
                st.session_state.rag_structure_profile_id
            ),
            hours_profile_snapshot=_profile_snapshot(
                st.session_state.rag_hours_profile_id
            ),
        )
        st.success("Estimación confirmada y guardada.")
        st.rerun()
