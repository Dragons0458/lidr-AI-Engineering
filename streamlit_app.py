import json
import os
from pathlib import Path
from streamlit.runtime.uploaded_file_manager import UploadedFile
import time

import httpx
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from app.schemas.acb import ACBResponse
from app.schemas.estimation import (
    DetailLevel,
    EstimationResponse,
    OutputFormat,
    ProjectType,
    ReferenceProject,
)
from streamlit_common import (
    env_display,
    format_api_error,
    render_structured_phases,
    resolve_sidebar_model,
)

TIER_OPTIONS = ["auto", "executive", "pm", "developer", "default"]

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_API_BASE_URL = "http://localhost:8000/api/v1"
REFERENCE_PROJECT_FIELDS = ("name", "summary", "estimated_hours", "team", "outcome")


def add_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)


def empty_project_metadata() -> dict[str, object]:
    return {
        "project_name": None,
        "assumed_team_size": None,
        "mentioned_technologies": [],
        "excluded_technologies": [],
        "agreed_scope": None,
    }


def get_api_base_url() -> str:
    env_url = os.getenv("ESTIMATION_API_BASE_URL", DEFAULT_API_BASE_URL)
    try:
        return str(st.secrets.get("ESTIMATION_API_BASE_URL", env_url))
    except StreamlitSecretNotFoundError:
        return env_url


def build_sessions_url() -> str:
    return f"{get_api_base_url().rstrip('/')}/sessions"


def build_session_estimate_url(session_id: str) -> str:
    return f"{build_sessions_url()}/{session_id}/estimate"


def build_session_estimate_acb_url(session_id: str) -> str:
    return f"{build_sessions_url()}/{session_id}/estimate-acb"


def build_session_debug_url(session_id: str) -> str:
    return f"{build_sessions_url()}/{session_id}"


def fetch_session_debug(session_id: str) -> dict[str, object]:
    response = httpx.get(build_session_debug_url(session_id), timeout=10.0)
    response.raise_for_status()
    return response.json()


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
        has_any_value = bool(
            name or summary or team or outcome or estimated_hours_raw not in (None, "")
        )

        if not has_any_value:
            continue

        if not all([name, summary, team, outcome]) or estimated_hours_raw in (None, ""):
            raise ValueError(
                f"Subproyecto {index}: complete name, summary, estimated_hours, team and outcome."
            )

        try:
            estimated_hours = int(estimated_hours_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Subproyecto {index}: estimated_hours debe ser un entero."
            ) from exc

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


def create_session() -> str:
    response = httpx.post(build_sessions_url(), json={}, timeout=20.0)
    response.raise_for_status()
    return str(response.json()["session_id"])


def reset_conversation_state() -> None:
    st.session_state.session_id = create_session()
    st.session_state.project_metadata = empty_project_metadata()
    st.session_state.model = "-"
    st.session_state.provider = "-"
    st.session_state.cache_hit = False
    st.session_state.out_of_scope = False
    st.session_state.cost_usd = 0.0
    st.session_state.prompt_version = "v1"
    st.session_state.response_time = 0.0
    st.session_state.last_estimation = ""
    st.session_state.last_error = ""
    st.session_state.messages = []
    st.session_state.attachments_locked = False
    st.session_state.reference_projects_rows = []
    st.session_state.use_acb = False
    st.session_state.last_acb_trace = None
    st.session_state.session_debug = {}
    st.session_state.form_version += 1


def clear_local_conversation_view() -> None:
    st.session_state.last_estimation = ""
    st.session_state.last_error = ""
    st.session_state.response_time = 0.0
    st.session_state.model = "-"
    st.session_state.provider = "-"
    st.session_state.cache_hit = False
    st.session_state.out_of_scope = False
    st.session_state.cost_usd = 0.0
    st.session_state.messages = []


def build_user_message_content(description: str, filenames: list[str]) -> str:
    normalized_description = description.strip()
    if normalized_description and filenames:
        files_text = "\n".join(f"- {filename}" for filename in filenames)
        return f"{normalized_description}\n\nArchivos adjuntos:\n{files_text}"
    if filenames:
        files_text = "\n".join(f"- {filename}" for filename in filenames)
        return f"Estimar el proyecto con base en estos adjuntos:\n{files_text}"
    return normalized_description


def is_retryable_estimation(estimation: str, *, out_of_scope: bool = False) -> bool:
    """True when the user should refine input (out of scope or legacy ambiguous text)."""
    if out_of_scope:
        return True
    normalized_estimation = estimation.lower()
    if normalized_estimation.startswith("out of scope:"):
        return True
    ambiguity_markers = (
        "demasiado ambiguo",
        "demasiado vaga",
        "requerimiento es ambiguo",
        "requerimiento es demasiado ambiguo",
        "no se puede generar una estimación",
    )
    return any(marker in normalized_estimation for marker in ambiguity_markers)


def apply_acb_response(response: ACBResponse) -> None:
    st.session_state.last_estimation = response.result.summary
    st.session_state.model = response.model
    st.session_state.provider = response.provider
    st.session_state.prompt_version = response.prompt_version
    st.session_state.cache_hit = response.cache_hit
    st.session_state.out_of_scope = response.result.summary.startswith("Out of scope:")
    st.session_state.cost_usd = response.cost_usd
    st.session_state.project_metadata = (
        response.project_metadata or empty_project_metadata()
    )
    st.session_state.last_acb_trace = response.acb.model_dump()
    st.session_state.last_structured_result = response.result.model_dump()


def refresh_session_debug() -> None:
    session_id = st.session_state.get("session_id")
    if not session_id:
        return
    try:
        st.session_state.session_debug = fetch_session_debug(session_id)
    except httpx.HTTPError:
        st.session_state.session_debug = {}


def apply_estimation_response(response: EstimationResponse) -> None:
    st.session_state.last_estimation = response.estimation
    st.session_state.model = response.model
    st.session_state.provider = response.provider
    st.session_state.prompt_version = response.prompt_version
    st.session_state.cache_hit = response.cache_hit
    st.session_state.out_of_scope = response.out_of_scope
    st.session_state.cost_usd = response.cost_usd
    st.session_state.project_metadata = (
        response.project_metadata or empty_project_metadata()
    )


def handle_api_failure(exc: Exception, *, api_base_url: str) -> str:
    if isinstance(exc, httpx.HTTPError):
        return format_api_error(exc, api_base_url=api_base_url)
    return f"**Error inesperado**\n\n{exc}"


if "model" not in st.session_state:
    st.session_state.model = "-"
if "provider" not in st.session_state:
    st.session_state.provider = "-"
if "cache_hit" not in st.session_state:
    st.session_state.cache_hit = False
if "out_of_scope" not in st.session_state:
    st.session_state.out_of_scope = False
if "cost_usd" not in st.session_state:
    st.session_state.cost_usd = 0.0
if "response_time" not in st.session_state:
    st.session_state.response_time = 0.0
if "last_estimation" not in st.session_state:
    st.session_state.last_estimation = ""
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "prompt_version" not in st.session_state:
    st.session_state.prompt_version = "v1"
if "project_metadata" not in st.session_state:
    st.session_state.project_metadata = empty_project_metadata()
if "reference_projects_rows" not in st.session_state:
    st.session_state.reference_projects_rows = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "attachments_locked" not in st.session_state:
    st.session_state.attachments_locked = False
if "form_version" not in st.session_state:
    st.session_state.form_version = 0
if "use_acb" not in st.session_state:
    st.session_state.use_acb = False
if "last_acb_trace" not in st.session_state:
    st.session_state.last_acb_trace = None
if "last_structured_result" not in st.session_state:
    st.session_state.last_structured_result = None
if "session_debug" not in st.session_state:
    st.session_state.session_debug = {}
if "tier_choice" not in st.session_state:
    st.session_state.tier_choice = "auto"
if "session_id" not in st.session_state:
    try:
        st.session_state.session_id = create_session()
    except httpx.HTTPError as exc:
        st.session_state.session_id = ""
        st.session_state.last_error = format_api_error(
            exc, api_base_url=get_api_base_url()
        )
    except Exception as exc:
        st.session_state.session_id = ""
        st.session_state.last_error = f"No se pudo crear la sesión: {exc}"

st.title("Estimador CAG")

with st.sidebar:
    st.title("Estimaciones")
    if st.button("Nueva conversación"):
        try:
            reset_conversation_state()
        except httpx.HTTPError as exc:
            st.session_state.last_error = format_api_error(
                exc, api_base_url=get_api_base_url()
            )
        except Exception as exc:
            st.session_state.last_error = f"No se pudo crear la nueva sesión: {exc}"
        st.rerun()
    if st.button("Limpiar vista local"):
        clear_local_conversation_view()
        st.rerun()

conversation_started = st.session_state.attachments_locked
form_title = "Nueva estimación" if not conversation_started else "Configuración"

with st.expander(form_title, expanded=not conversation_started):
    with st.form("estimation_form"):
        if conversation_started:
            st.caption(
                "La conversación ya inició. Los nuevos mensajes se escriben en el chat inferior."
            )
            transcription = ""
            uploaded_files = []
        else:
            transcription = st.text_area(
                "Descripción o instrucciones",
                placeholder=(
                    "Describe el proyecto, pega un resumen, o déjalo vacío si los "
                    "archivos adjuntos contienen el alcance."
                ),
                height=200,
                key=f"transcription_{st.session_state.form_version}",
                value="Una app para hacer tortas de queso",
            )
            uploaded_files = st.file_uploader(
                "Archivos",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key=f"uploaded_files_{st.session_state.form_version}",
            )
        use_acb = st.checkbox(
            "Usar Actor-Critic-Boss (estructurado)",
            value=st.session_state.use_acb,
            help="Llama a POST .../estimate-acb (salida estructurada + traza).",
        )
        tier_choice = st.selectbox(
            "Audiencia (tier)",
            options=TIER_OPTIONS,
            index=TIER_OPTIONS.index(st.session_state.tier_choice),
            help="auto deja que el servidor resuelva cuando TIER_RESOLUTION_ENABLED=true.",
        )
        prompt_version = st.selectbox(
            "Versión de prompt",
            options=["v1", "v2"] if not use_acb else ["v3"],
            index=0 if use_acb else ["v1", "v2"].index(st.session_state.prompt_version),
        )
        project_type = st.selectbox(
            "Tipo de proyecto",
            options=list[ProjectType](ProjectType),
            format_func=lambda value: value.value,
        )
        detail_level = st.selectbox(
            "Nivel de detalle",
            options=list[DetailLevel](DetailLevel),
            format_func=lambda value: value.value,
        )
        output_format = st.selectbox(
            "Formato de salida",
            options=list[OutputFormat](OutputFormat),
            format_func=lambda value: value.value,
        )
        reference_projects_rows = st.session_state.reference_projects_rows
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
                key=f"reference_projects_{st.session_state.form_version}",
                column_config={
                    "name": st.column_config.TextColumn(
                        "name", required=False, width="small"
                    ),
                    "summary": st.column_config.TextColumn(
                        "summary", required=False, width="large"
                    ),
                    "estimated_hours": st.column_config.NumberColumn(
                        "estimated_hours",
                        required=False,
                        min_value=1,
                        step=1,
                    ),
                    "team": st.column_config.TextColumn(
                        "team", required=False, width="medium"
                    ),
                    "outcome": st.column_config.TextColumn(
                        "outcome", required=False, width="medium"
                    ),
                },
            )
        submit_label = (
            "Generar estimación"
            if not conversation_started
            else "Guardar configuración"
        )
        submitted = st.form_submit_button(
            submit_label, disabled=not st.session_state.session_id
        )

st.session_state.prompt_version = prompt_version
st.session_state.use_acb = use_acb
st.session_state.tier_choice = tier_choice

if submitted:
    start_time = time.perf_counter()
    st.session_state.last_error = ""
    st.session_state.prompt_version = prompt_version

    try:
        reference_projects: list[ReferenceProject] | None = None
        if prompt_version == "v2":
            rows_as_dicts = _table_rows_to_dicts(reference_projects_rows)
            st.session_state.reference_projects_rows = [
                row
                for row in rows_as_dicts
                if any(
                    str(row.get(field, "")).strip()
                    for field in REFERENCE_PROJECT_FIELDS
                    if field != "estimated_hours"
                )
                or row.get("estimated_hours") not in (None, "")
            ]
            reference_projects = _parse_reference_projects(rows_as_dicts) or None

        if conversation_started:
            st.rerun()

        submitted_files = list[UploadedFile](uploaded_files)
        if not transcription.strip() and not submitted_files:
            st.session_state.last_error = (
                "Agrega una descripción o adjunta al menos un PDF/DOCX con el alcance."
            )
            st.rerun()

        files = [
            (
                "attachments",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                ),
            )
            for uploaded_file in submitted_files
        ]
        user_message = build_user_message_content(
            transcription, [uploaded_file.name for uploaded_file in submitted_files]
        )
        data = {
            "description": transcription,
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        }
        if reference_projects:
            data["reference_projects"] = json.dumps(
                [
                    reference_project.model_dump(mode="json")
                    for reference_project in reference_projects
                ]
            )
        if tier_choice != "auto":
            data["tier"] = tier_choice

        estimate_url = (
            build_session_estimate_acb_url(st.session_state.session_id)
            if use_acb
            else build_session_estimate_url(st.session_state.session_id)
        )
        response = httpx.post(
            estimate_url,
            data=data,
            files=files,
            params={"prompt_version": prompt_version},
            timeout=120.0,
        )
        response.raise_for_status()

        if use_acb:
            acb_response = ACBResponse.model_validate(response.json())
            apply_acb_response(acb_response)
            assistant_text = acb_response.result.summary
            needs_retry = is_retryable_estimation(
                assistant_text,
                out_of_scope=st.session_state.out_of_scope,
            )
        else:
            estimation_response = EstimationResponse.model_validate(response.json())
            apply_estimation_response(estimation_response)
            assistant_text = estimation_response.estimation
            needs_retry = is_retryable_estimation(
                assistant_text,
                out_of_scope=estimation_response.out_of_scope,
            )
        refresh_session_debug()
        st.session_state.messages.append({"role": "user", "content": user_message})
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )
        st.session_state.attachments_locked = not needs_retry
        if needs_retry:
            if st.session_state.out_of_scope:
                st.session_state.last_error = (
                    "La solicitud quedó **fuera de alcance** para este estimador "
                    "(solo proyectos de software). Amplía la descripción con requisitos "
                    "técnicos o cambia el tipo de proyecto."
                )
            else:
                st.session_state.last_error = (
                    "La estimación no encontró alcance suficiente. Puedes ampliar la "
                    "descripción o adjuntar otro PDF/DOCX."
                )
    except httpx.HTTPError as exc:
        st.session_state.last_estimation = ""
        friendly_error = handle_api_failure(exc, api_base_url=get_api_base_url())
        st.session_state.last_error = friendly_error
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly_error}
        )
    except Exception as exc:
        st.session_state.last_estimation = ""
        friendly_error = handle_api_failure(exc, api_base_url=get_api_base_url())
        st.session_state.last_error = friendly_error
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly_error}
        )
    finally:
        st.session_state.response_time = time.perf_counter() - start_time
        st.session_state.form_version += 1
        st.rerun()

if st.session_state.last_error and not st.session_state.messages:
    st.error(st.session_state.last_error)

for message in st.session_state.messages:
    add_message(message["role"], message["content"])

chat_disabled = (
    not st.session_state.session_id or not st.session_state.attachments_locked
)
chat_placeholder = (
    "Primero genera una estimación desde el formulario superior"
    if not st.session_state.attachments_locked
    else "Escribe un nuevo mensaje"
)

if prompt := st.chat_input(chat_placeholder, disabled=chat_disabled):
    start_time = time.perf_counter()
    st.session_state.last_error = ""
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        data = {
            "description": prompt,
            "project_type": project_type.value,
            "detail_level": detail_level.value,
            "output_format": output_format.value,
        }
        if st.session_state.tier_choice != "auto":
            data["tier"] = st.session_state.tier_choice
        estimate_url = (
            build_session_estimate_acb_url(st.session_state.session_id)
            if st.session_state.use_acb
            else build_session_estimate_url(st.session_state.session_id)
        )
        response = httpx.post(
            estimate_url,
            data=data,
            files=[],
            params={"prompt_version": st.session_state.prompt_version},
            timeout=120.0,
        )
        response.raise_for_status()

        if st.session_state.use_acb:
            acb_response = ACBResponse.model_validate(response.json())
            apply_acb_response(acb_response)
            assistant_text = acb_response.result.summary
            out_of_scope = st.session_state.out_of_scope
        else:
            estimation_response = EstimationResponse.model_validate(response.json())
            apply_estimation_response(estimation_response)
            assistant_text = estimation_response.estimation
            out_of_scope = estimation_response.out_of_scope
        refresh_session_debug()
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_text}
        )
        st.session_state.attachments_locked = not is_retryable_estimation(
            assistant_text,
            out_of_scope=out_of_scope,
        )
    except httpx.HTTPError as exc:
        st.session_state.last_estimation = ""
        friendly_error = handle_api_failure(exc, api_base_url=get_api_base_url())
        st.session_state.last_error = friendly_error
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly_error}
        )
    except Exception as exc:
        st.session_state.last_estimation = ""
        friendly_error = handle_api_failure(exc, api_base_url=get_api_base_url())
        st.session_state.last_error = friendly_error
        st.session_state.messages.append(
            {"role": "assistant", "content": friendly_error}
        )
    finally:
        st.session_state.response_time = time.perf_counter() - start_time
        st.rerun()

with st.sidebar:
    st.header("Configuración")
    st.text_input("API base URL", value=get_api_base_url(), disabled=True)
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    st.divider()
    st.subheader("Última respuesta")
    st.metric(
        "Model",
        resolve_sidebar_model(response_model=st.session_state.model),
    )
    if st.session_state.provider and st.session_state.provider != "-":
        st.caption(f"Provider: `{st.session_state.provider}`")
    st.metric("Prompt version", st.session_state.prompt_version)
    st.metric("Response time", f"{st.session_state.response_time:.2f}s")
    if st.session_state.cost_usd:
        st.metric("Cost (USD)", f"{st.session_state.cost_usd:.4f}")
    st.metric("Cache hit", "Sí" if st.session_state.cache_hit else "No")
    if st.session_state.out_of_scope:
        st.warning("Fuera de alcance (out_of_scope)")
    st.divider()
    st.subheader("Modelo configurado (.env)")
    st.text_input("LLM_PROVIDER", value=env_display("LLM_PROVIDER"), disabled=True)
    st.text_input("PRIMARY_MODEL", value=env_display("PRIMARY_MODEL"), disabled=True)
    st.text_input(
        "FALLBACK_MODEL",
        value=env_display("FALLBACK_MODEL", "(sin fallback)"),
        disabled=True,
    )
    debug = st.session_state.session_debug or {}
    if debug:
        st.metric("Anclas", debug.get("anchors_count", 0))
        st.metric("Resumen (chars)", debug.get("summary_chars", 0))
        if debug.get("last_resolved_tier"):
            st.caption(
                f"Tier: `{debug.get('last_resolved_tier')}` "
                f"({debug.get('last_tier_rule')})"
            )
    if st.session_state.last_structured_result:
        with st.expander("Estimación estructurada (ACB)", expanded=True):
            render_structured_phases(st.session_state.last_structured_result)
    if st.session_state.last_acb_trace:
        with st.expander("Traza ACB", expanded=False):
            st.json(st.session_state.last_acb_trace)
    with st.expander("project_metadata", expanded=False):
        st.json(st.session_state.project_metadata)
    with st.expander("Sesión 5 (.env)", expanded=False):
        st.text_input(
            "TIER_RESOLUTION_ENABLED",
            value=env_display("TIER_RESOLUTION_ENABLED", "false"),
            disabled=True,
        )
        st.text_input(
            "MEMORY_COMPRESSION_ENABLED",
            value=env_display("MEMORY_COMPRESSION_ENABLED", "false"),
            disabled=True,
        )
