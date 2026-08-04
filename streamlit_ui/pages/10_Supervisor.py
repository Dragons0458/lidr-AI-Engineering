"""Session 14 supervisor multi-agent wizard (routing + privilege + HITL)."""

from __future__ import annotations

import runpy
import uuid
from datetime import datetime
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import pandas as pd
import streamlit as st

from streamlit_ui.common import get_api_root_url, get_estimate_api_key
from streamlit_ui.store import (
    apply_supervisor_run_state,
    create_supervisor_run,
    get_supervisor_run,
    list_supervisor_runs,
    list_supervisor_runs_awaiting,
)
from streamlit_ui.supervisor_flow import (
    competition_summary,
    confidence_pct,
    deferred_rows,
    irreversible_rows,
    load_sample_transcript,
    status_badge_label,
    supervisor_resume,
    supervisor_start,
    supervisor_state,
)

st.set_page_config(page_title="Supervisor", page_icon="🧭", layout="wide")
st.title("Supervisor multiagente (Sesión 14)")
st.caption(
    "Estrella con router LLM, privilegio exigible, competición y sandbox de escritura."
)

api_root = get_api_root_url()
api_key = get_estimate_api_key()

if "supervisor_state" not in st.session_state:
    st.session_state.supervisor_state = None
if "supervisor_estimation_id" not in st.session_state:
    st.session_state.supervisor_estimation_id = None


def _badge(status: str | None) -> None:
    label = status_badge_label(status)
    st.markdown(f"### Estado: `{label}`")


def _age_label(created_at: str | None) -> str:
    if not created_at:
        return "—"
    try:
        created = datetime.fromisoformat(created_at)
        delta = datetime.utcnow() - created.replace(tzinfo=None)
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        if hours < 48:
            return f"{hours}h"
        return f"{hours // 24}d"
    except ValueError:
        return created_at


tab_inbox, tab_start, tab_review, tab_done, tab_recover = st.tabs(
    [
        "0. Bandeja",
        "1. Inicio",
        "2. Revisión humana",
        "3. Completado",
        "4. Recuperar run",
    ]
)

with tab_inbox:
    st.subheader("Pendientes de revisión")
    awaiting = list_supervisor_runs_awaiting(limit=50)
    if awaiting:
        rows = []
        for run in awaiting:
            pending = run.get("pending_review") or {}
            reasons = pending.get("reasons") or []
            rows.append(
                {
                    "estimation_id": run["estimation_id"],
                    "reason": reasons[0] if reasons else "—",
                    "confidence_%": confidence_pct(
                        pending.get("confidence") or run.get("confidence")
                    ),
                    "age": _age_label(run.get("created_at")),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for run in awaiting:
            if st.button(
                f"Abrir {run['estimation_id']}",
                key=f"await_{run['estimation_id']}",
            ):
                st.session_state.supervisor_estimation_id = run["estimation_id"]
                try:
                    remote = supervisor_state(
                        run["estimation_id"],
                        api_root=api_root,
                        api_key=api_key,
                    )
                    apply_supervisor_run_state(run["estimation_id"], remote)
                    st.session_state.supervisor_state = remote
                except Exception:  # noqa: BLE001 — fall back to mirror
                    st.session_state.supervisor_state = {
                        "estimation_id": run["estimation_id"],
                        "state": run.get("run_state"),
                        "status": run.get("status"),
                        "pending_review": run.get("pending_review"),
                        "estimate": run.get("estimate"),
                        "confidence": run.get("confidence"),
                        "divergence": run.get("divergence"),
                        "synthesis": run.get("synthesis"),
                        "proposals": run.get("proposals") or [],
                        "saved": run.get("saved"),
                        "agent_contributions": run.get("agent_contributions") or [],
                    }
                st.rerun()
    else:
        st.caption("No hay runs en awaiting_human_review.")

    st.subheader("Recientes")
    recent = list_supervisor_runs(limit=20)
    if recent:
        recent_rows = [
            {
                "estimation_id": run["estimation_id"],
                "badge": status_badge_label(run.get("status")),
                "total_hours": run.get("total_hours"),
                "updated_at": run.get("updated_at"),
            }
            for run in recent
        ]
        st.dataframe(
            pd.DataFrame(recent_rows), use_container_width=True, hide_index=True
        )
        for run in recent:
            if st.button(
                f"Cargar {run['estimation_id']}",
                key=f"recent_{run['estimation_id']}",
            ):
                st.session_state.supervisor_estimation_id = run["estimation_id"]
                try:
                    remote = supervisor_state(
                        run["estimation_id"],
                        api_root=api_root,
                        api_key=api_key,
                    )
                    apply_supervisor_run_state(run["estimation_id"], remote)
                    st.session_state.supervisor_state = remote
                except Exception:  # noqa: BLE001
                    mirrored = get_supervisor_run(run["estimation_id"]) or run
                    st.session_state.supervisor_state = {
                        "estimation_id": mirrored["estimation_id"],
                        "state": mirrored.get("run_state"),
                        "status": mirrored.get("status"),
                        "pending_review": mirrored.get("pending_review"),
                        "estimate": mirrored.get("estimate"),
                        "confidence": mirrored.get("confidence"),
                        "divergence": mirrored.get("divergence"),
                        "synthesis": mirrored.get("synthesis"),
                        "proposals": mirrored.get("proposals") or [],
                        "saved": mirrored.get("saved"),
                        "agent_contributions": mirrored.get("agent_contributions")
                        or [],
                    }
                st.rerun()
    else:
        st.caption("Sin runs recientes.")

with tab_start:
    sample = st.selectbox(
        "Cargar transcripción de ejemplo",
        [
            "(ninguna)",
            "happy_path",
            "edge_case",
            "low_confidence",
            "no_precedent",
            "out_of_historical_range",
        ],
    )
    default_text = ""
    if sample != "(ninguna)":
        try:
            default_text = load_sample_transcript(sample)
        except FileNotFoundError:
            st.warning(f"No se encontró la muestra `{sample}`.")

    transcript = st.text_area("Transcripción", value=default_text, height=260)
    if st.button(
        "Arrancar supervisor", type="primary", disabled=len(transcript.strip()) < 100
    ):
        estimation_id = f"st-s14-{uuid.uuid4()}"
        try:
            create_supervisor_run(estimation_id, transcript.strip())
            state = supervisor_start(
                transcript.strip(),
                estimation_id=estimation_id,
                api_root=api_root,
                api_key=api_key,
            )
            apply_supervisor_run_state(estimation_id, state)
            st.session_state.supervisor_state = state
            st.session_state.supervisor_estimation_id = state["estimation_id"]
            st.success(f"Run: `{state['estimation_id']}`")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error al arrancar: {exc}")

with tab_review:
    state = st.session_state.supervisor_state
    if not state:
        st.info("Arranca un run o recupéralo desde la bandeja / pestaña 4.")
    else:
        _badge(state.get("status"))
        pending = state.get("pending_review")
        if state.get("status") == "awaiting_human_review" and pending:
            st.write("**Razones:**")
            for reason in pending.get("reasons") or []:
                st.write(f"- `{reason}`")
            risk_flags = pending.get("risk_flags") or []
            if risk_flags:
                st.write("**Señales de riesgo:**")
                for flag in risk_flags:
                    st.write(f"- `{flag}`")

            pct = confidence_pct(pending.get("confidence"))
            if pct is not None:
                st.write(f"Confianza: **{pct}%** (umbral `{pending.get('threshold')}`)")

            divergence = pending.get("divergence") or state.get("divergence")
            summary = competition_summary(
                {
                    **state,
                    "divergence": divergence,
                    "synthesis": pending.get("synthesis") or state.get("synthesis"),
                    "estimate": pending.get("estimate") or state.get("estimate"),
                }
            )
            if summary:
                st.info(
                    f"Divergencia: level=`{summary.get('level')}` "
                    f"ratio=`{summary.get('ratio')}` "
                    f"spread=`{summary.get('spread')}` · "
                    f"rango sintetizado `{summary.get('low')}–{summary.get('high')}` h"
                )

            if pending.get("persist_requested") or state.get("persist_requested"):
                st.warning(
                    "Aprobar autoriza una escritura **IRREVERSIBLE** (`save_estimate`)."
                )

            estimate = pending.get("estimate") or state.get("estimate") or {}
            components = list(estimate.get("components") or [])
            edited_components: list[dict] = []
            if components:
                st.write("**Horas por componente**")
                for index, component in enumerate(components):
                    hours = st.number_input(
                        f"{component.get('name', f'component-{index}')}",
                        min_value=0.0,
                        value=float(component.get("estimated_hours") or 0.0),
                        step=1.0,
                        key=f"comp_hours_{index}",
                    )
                    edited_components.append(
                        {**component, "estimated_hours": float(hours)}
                    )
            else:
                st.json(estimate)

            note = st.text_input("Nota (opcional)")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                approve = st.button("Aprobar", type="primary")
            with col_b:
                adjust = st.button("Ajustar y aprobar")
            with col_c:
                reject = st.button("Rechazar")

            action = None
            overrides = None
            if approve:
                action = "approve"
            elif adjust:
                action = "adjust"
                overrides = {"components": edited_components or components}
            elif reject:
                action = "reject"

            if action:
                try:
                    updated = supervisor_resume(
                        state["estimation_id"],
                        action,
                        estimate_overrides=overrides,
                        note=note or None,
                        api_root=api_root,
                        api_key=api_key,
                    )
                    apply_supervisor_run_state(state["estimation_id"], updated)
                    st.session_state.supervisor_state = updated
                    st.success("Reanudado.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Error al reanudar: {exc}")
        elif state.get("status") == "needs_review":
            st.info(
                "`needs_review` es una advertencia del validador de coherencia "
                "(issues / fuera de banda), **no** la pausa HITL. "
                "El gate humano solo aparece con `awaiting_human_review`. "
                "Mira la pestaña **3. Completado** para el resultado."
            )
        else:
            st.write("No hay revisión humana pendiente.")

with tab_done:
    state = st.session_state.supervisor_state
    if not state:
        st.info("Sin run en sesión.")
    else:
        _badge(state.get("status"))
        pct = confidence_pct(state.get("confidence"))
        if pct is not None:
            st.write(f"Confianza: **{pct}%**")

        summary = competition_summary(state)
        if summary:
            st.subheader("Rango (competición conservador ↔ agresivo)")
            st.write(f"`{summary.get('low')}–{summary.get('high')}` horas")
            questions = summary.get("open_questions") or []
            if questions:
                st.write("Preguntas abiertas:")
                for question in questions:
                    st.write(f"- {question}")

        proposals = state.get("proposals") or []
        if proposals:
            st.subheader("Propuestas")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "stance": p.get("stance"),
                            "total_hours": p.get("total_hours"),
                            "risks": "; ".join(p.get("risks") or []),
                        }
                        for p in proposals
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        if state.get("estimate"):
            st.subheader("Estimación")
            st.json(state["estimate"])

        saved = state.get("saved")
        if saved is not None:
            if saved.get("ok"):
                st.success("Persistencia: persistida")
            else:
                st.warning(
                    f"Persistencia: NO persistida "
                    f"({saved.get('error') or saved.get('summary') or 'motivo desconocido'})"
                )

        st.subheader("Historial de enrutado")
        routing = state.get("routing_history") or []
        if routing:
            st.dataframe(pd.DataFrame(routing), use_container_width=True)
        else:
            st.caption("Sin filas de enrutado.")

        st.subheader("Auditoría de acciones")
        contributions = state.get("agent_contributions") or []
        if contributions:
            display_rows = []
            for row in contributions:
                outcome = row.get("outcome")
                badge = ""
                if outcome == "deferred":
                    badge = "DIFERIDA"
                if row.get("tool") == "save_estimate":
                    badge = (badge + " · " if badge else "") + "IRREVERSIBLE"
                display_rows.append({**row, "badge": badge or "—"})
            df = pd.DataFrame(display_rows)
            st.dataframe(df, use_container_width=True)
            deferred = deferred_rows(contributions)
            if deferred:
                st.markdown(
                    f"<div style='background:#fef3c7;padding:0.5rem;border-radius:4px'>"
                    f"{len(deferred)} acción(es) diferida(s)</div>",
                    unsafe_allow_html=True,
                )
            irreversible = irreversible_rows(contributions)
            if irreversible:
                st.caption(
                    f"{len(irreversible)} fila(s) con tool `save_estimate` (IRREVERSIBLE)."
                )
            violations = state.get("privilege_violations") or []
            if violations:
                st.warning(
                    f"{len(violations)} denegación(es) de privilegio en el trail."
                )
        else:
            st.caption("Sin contribuciones.")

with tab_recover:
    estimation_id = st.text_input(
        "estimation_id",
        value=st.session_state.supervisor_estimation_id or "",
    )
    if st.button("Cargar estado", disabled=not estimation_id.strip()):
        try:
            state = supervisor_state(
                estimation_id.strip(),
                api_root=api_root,
                api_key=api_key,
            )
            if get_supervisor_run(state["estimation_id"]) is None:
                create_supervisor_run(
                    state["estimation_id"],
                    transcript="(recovered without transcript)",
                )
            apply_supervisor_run_state(state["estimation_id"], state)
            st.session_state.supervisor_state = state
            st.session_state.supervisor_estimation_id = state["estimation_id"]
            st.success("Estado cargado.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo cargar: {exc}")
