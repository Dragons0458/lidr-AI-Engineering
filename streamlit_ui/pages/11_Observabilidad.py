"""Session 16 — production observability dashboard.

Vital signs of the live AI service: latency (mean + p95), cost per request,
error rate, abstention, cache hits. Talks to ``ai-service`` over HTTP — never
to Postgres directly (Session 15 public/private frontier).
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "path_setup.py"))

import pandas as pd
import streamlit as st

from streamlit_ui.common import (
    fetch_observability_metrics,
    fetch_observability_requests,
    format_api_error,
    get_api_base_url,
    get_api_root_url,
    load_latest_eval_report,
)

st.set_page_config(page_title="Observabilidad", page_icon="📈", layout="wide")
st.title("Observabilidad")
st.caption(
    "Constantes vitales de producción: latencia, coste y errores de las "
    "peticiones de estimación. Esto no es el golden set (laboratorio)."
)

api_root = get_api_root_url(get_api_base_url())

with st.sidebar:
    st.header("Ventana")
    window_label = st.radio("Periodo", ["1h", "24h", "7d"], index=1)
    window_hours = {"1h": 1, "24h": 24, "7d": 168}[window_label]
    route_filter = (
        st.text_input(
            "Filtro de ruta (opcional)",
            placeholder="/v1/estimate/from-transcript",
        ).strip()
        or None
    )

try:
    summary = fetch_observability_metrics(
        api_root,
        window_hours,
        route=route_filter,
    )
except Exception as exc:
    st.error(format_api_error(exc, api_base_url=get_api_base_url()))
    st.stop()

latency = summary.get("latency_ms") or {}
cost = summary.get("cost_usd") or {}
n = int(latency.get("n") or summary.get("requests") or 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latencia media", f"{float(latency.get('mean') or 0):.0f} ms")
c2.metric(
    "Latencia p95",
    f"{float(latency.get('p95') or 0):.0f} ms",
    help=f"n={n}. Un p95 con pocas muestras no significa nada.",
)
c2.caption(f"n = {n}")
c3.metric("Coste medio / petición", f"${float(cost.get('mean_per_request') or 0):.4f}")
c4.metric("Tasa de error", f"{float(summary.get('error_rate') or 0):.1%}")

st.subheader("Series temporales")
series = summary.get("series") or []
if series:
    frame = pd.DataFrame(series)
    if "bucket" in frame.columns:
        frame = frame.set_index("bucket")
    cols = [
        col
        for col in ("requests", "p95_latency_ms", "cost_usd")
        if col in frame.columns
    ]
    if cols:
        st.line_chart(frame[cols])
else:
    st.info("Aún no hay buckets en esta ventana.")

st.subheader("Señales de calidad")
q1, q2 = st.columns(2)
q1.metric("Tasa de abstención", f"{float(summary.get('abstention_rate') or 0):.1%}")
q1.caption(
    "Si la abstención sube de golpe, algo ha cambiado en los datos, no en el código."
)
q2.metric("Cache hit rate", f"{float(summary.get('cache_hit_rate') or 0):.1%}")
q2.caption(
    "Un hit-rate alto en una eval suele significar que estás midiendo caché, no al modelo."
)

st.subheader("Últimas peticiones")
st.caption("El `request_id` de una fila lenta es lo que se busca en Logfire.")
try:
    rows = fetch_observability_requests(api_root, limit=50)
except Exception as exc:
    st.warning(format_api_error(exc, api_base_url=get_api_base_url()))
    rows = []
if rows:
    st.dataframe(
        [
            {
                "request_id": row.get("request_id"),
                "ruta": row.get("route"),
                "estado": row.get("status"),
                "latencia_ms": row.get("latency_ms"),
                "coste_usd": row.get("estimated_cost_usd"),
                "confianza": row.get("confidence"),
            }
            for row in rows
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No hay filas instrumentadas todavía. Lanza una estimación o el golden set."
    )

lab = load_latest_eval_report()
if lab:
    with st.expander("Laboratorio vs producción"):
        st.warning(
            "Esto es el último informe del golden set (laboratorio), no tráfico de producción."
        )
        arms = lab.get("arms") or {}
        cols = st.columns(max(len(arms), 1))
        for index, (name, arm) in enumerate(arms.items()):
            rate = (arm or {}).get("within_range_rate")
            label = f"{rate:.0%}" if isinstance(rate, (int, float)) else "—"
            cols[index].metric(f"within_range_rate [{name}]", label)
        st.caption(f"run_id `{lab.get('run_id')}` · label `{lab.get('label')}`")
