import os
import time

import httpx
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)

DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"
REFERENCE_PROJECT_FIELDS = ("name", "summary", "estimated_hours", "team", "outcome")


def get_api_base_url() -> str:
    env_url = os.getenv("ESTIMATION_API_BASE_URL", DEFAULT_API_BASE_URL)
    try:
        return str(st.secrets.get("ESTIMATION_API_BASE_URL", env_url))
    except StreamlitSecretNotFoundError:
        return env_url


def build_estimate_url() -> str:
    return f"{get_api_base_url().rstrip('/')}/estimate"


def _table_rows_to_dicts(raw_rows: object) -> list[dict[str, object]]:
    if hasattr(raw_rows, "to_dict"):
        return raw_rows.to_dict(orient="records")
    if isinstance(raw_rows, list):
        return [row for row in raw_rows if isinstance(row, dict)]
    return []


def _parse_reference_projects(rows: list[dict[str, object]]) -> list[ReferenceProject]:
    reference_projects: list[ReferenceProject] = []

    for index, row in enumerate(rows, start=1):
        name = str(row.get("name", "")).strip()
        summary = str(row.get("summary", "")).strip()
        team = str(row.get("team", "")).strip()
        outcome = str(row.get("outcome", "")).strip()
        estimated_hours_raw = row.get("estimated_hours")
        has_any_value = bool(name or summary or team or outcome or estimated_hours_raw not in (None, ""))

        if not has_any_value:
            continue

        if not all([name, summary, team, outcome]) or estimated_hours_raw in (None, ""):
            raise ValueError(
                f"Subproyecto {index}: complete name, summary, estimated_hours, team and outcome."
            )

        try:
            estimated_hours = int(estimated_hours_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Subproyecto {index}: estimated_hours debe ser un entero.") from exc

        reference_projects.append(
            ReferenceProject(
                name=name,
                summary=summary,
                estimated_hours=estimated_hours,
                team=team,
                outcome=outcome,
            )
        )

    return reference_projects


if "model" not in st.session_state:
    st.session_state.model = "-"
if "response_time" not in st.session_state:
    st.session_state.response_time = 0.0
if "last_estimation" not in st.session_state:
    st.session_state.last_estimation = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "prompt_version" not in st.session_state:
    st.session_state.prompt_version = "v1"
if "reference_projects_rows" not in st.session_state:
    st.session_state.reference_projects_rows = []

st.title("Estimador CAG")

prompt_version = st.selectbox(
    "Versión de prompt",
    options=["v1", "v2"],
    index=["v1", "v2"].index(st.session_state.prompt_version),
)
st.session_state.prompt_version = prompt_version

with st.form("estimation_form"):
    reference_projects_rows = st.session_state.reference_projects_rows
    transcript = st.text_area(
        "Resumen de la reunión",
        placeholder="Describe el alcance, funcionalidades, equipo, plazos y restricciones...",
        height=200,
    )
    project_type = st.selectbox(
        "Tipo de proyecto",
        options=list(ProjectType),
        format_func=lambda value: value.value,
    )
    detail_level = st.selectbox(
        "Nivel de detalle",
        options=list(DetailLevel),
        format_func=lambda value: value.value,
    )
    output_format = st.selectbox(
        "Formato de salida",
        options=list(OutputFormat),
        format_func=lambda value: value.value,
    )
    if prompt_version == "v2":
        reference_projects_table_seed = (
            st.session_state.reference_projects_rows
            if st.session_state.reference_projects_rows
            else [{field: "" for field in REFERENCE_PROJECT_FIELDS}]
        )
        reference_projects_table_seed[0]["estimated_hours"] = (
            reference_projects_table_seed[0].get("estimated_hours") or None
        )
        st.caption("Proyectos de referencia (opcional)")
        reference_projects_rows = st.data_editor(
            reference_projects_table_seed,
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("name", required=False, width="small"),
                "summary": st.column_config.TextColumn("summary", required=False, width="large"),
                "estimated_hours": st.column_config.NumberColumn(
                    "estimated_hours",
                    required=False,
                    min_value=1,
                    step=1,
                ),
                "team": st.column_config.TextColumn("team", required=False, width="medium"),
                "outcome": st.column_config.TextColumn("outcome", required=False, width="medium"),
            },
        )
    submitted = st.form_submit_button("Generar estimación")

if submitted:
    start_time = time.perf_counter()
    st.session_state.last_error = ""

    try:
        reference_projects: list[ReferenceProject] | None = None
        if prompt_version == "v2":
            rows_as_dicts = _table_rows_to_dicts(reference_projects_rows)
            st.session_state.reference_projects_rows = [
                row
                for row in rows_as_dicts
                if any(str(row.get(field, "")).strip() for field in REFERENCE_PROJECT_FIELDS if field != "estimated_hours")
                or row.get("estimated_hours") not in (None, "")
            ]
            reference_projects = _parse_reference_projects(rows_as_dicts) or None

        request_payload = EstimationRequest(
            transcript=transcript,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            reference_projects=reference_projects,
        )
        response = httpx.post(
            build_estimate_url(),
            json=request_payload.model_dump(mode="json"),
            params={"prompt_version": prompt_version},
            timeout=120.0,
        )
        response.raise_for_status()

        estimation_response = EstimationResponse.model_validate(response.json())
        st.session_state.last_estimation = estimation_response.estimation
        st.session_state.model = estimation_response.model
        st.session_state.prompt_version = estimation_response.prompt_version
    except Exception as exc:
        st.session_state.last_estimation = ""
        st.session_state.model = "-"
        st.session_state.last_error = f"Error consumiendo la API: {exc}"
    finally:
        st.session_state.response_time = time.perf_counter() - start_time

if st.session_state.last_error:
    st.error(st.session_state.last_error)
elif st.session_state.last_estimation:
    st.markdown(st.session_state.last_estimation)

with st.sidebar:
    st.title("Estimaciones")
    if st.button("Limpiar resultado"):
        st.session_state.last_estimation = ""
        st.session_state.last_error = ""
        st.session_state.response_time = 0.0
        st.session_state.model = "-"
        st.session_state.prompt_version = "v1"
        st.session_state.reference_projects_rows = []
        st.rerun()
    st.text_input("API base URL", value=get_api_base_url(), disabled=True)
    st.metric("Model", st.session_state.model)
    st.metric("Prompt version", st.session_state.prompt_version)
    st.metric("Response time", f"{st.session_state.response_time:.2f}s")
