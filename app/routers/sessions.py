import json
from json import JSONDecodeError
from typing import Annotated, Literal
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import TypeAdapter, ValidationError

from app.errors.llm_error import LLMServiceError
from app.formatters.llm_formatters import format_response
from app.schemas.estimation import (
    DetailLevel,
    ExampleFormat,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    PreprocessingMode,
    ProjectType,
    ReferenceProject,
)
from app.schemas.sessions import (
    SessionCreateRequest,
    SessionDebugResponse,
    SessionResponse,
)
from app.services.attachment_service import (
    AttachmentTextExtractionError,
    UnsupportedAttachmentTypeError,
    extract_attachment_texts,
)
from app.services.estimation_service import generate_estimation
from app.services.evaluation import evaluate_estimation_structure
from app.services.project_metadata_extractor import extract_project_metadata
from app.services.sessions import Session


log = structlog.get_logger()
router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest | None = None,
) -> SessionResponse:
    """Create an in-memory IA session and return its identifier."""
    session_id = str(uuid4())
    Session.get_or_create(session_id)
    return SessionResponse(session_id=session_id)


@router.get("/sessions/{session_id}", response_model=SessionDebugResponse)
async def get_session_debug(session_id: str) -> SessionDebugResponse:
    """Return the in-memory session snapshot used by stress-test metrics."""
    session = Session.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDebugResponse(
        session_id=session.session_id,
        message_count=sum(len(turn.messages) for turn in session.history.turns),
        anchors_count=0,
        summary_chars=0,
        summary="",
        anchors=[],
        metadata=session.metadata.model_dump(),
        last_turn_observation=session.last_turn_observation,
    )


@router.post("/sessions/{session_id}/estimate", response_model=EstimationResponse)
async def estimate_session(
    session_id: str,
    description: Annotated[str, Form()] = "",
    project_type: Annotated[ProjectType, Form()] = ProjectType.WEB_SAAS,
    detail_level: Annotated[DetailLevel, Form()] = DetailLevel.MEDIUM,
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.LINE_ITEMS,
    reference_projects: Annotated[str | None, Form()] = None,
    preprocessing: Annotated[PreprocessingMode, Form()] = "none",
    use_examples: Annotated[bool, Form()] = True,
    num_examples: Annotated[int, Form(ge=0, le=5)] = 3,
    example_format: Annotated[ExampleFormat, Form()] = "markdown",
    model: Annotated[str | None, Form()] = None,
    max_tokens: Annotated[int, Form(ge=256, le=16000)] = 4000,
    evaluate: Annotated[bool, Form()] = True,
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    prompt_version: Literal["v1", "v2"] = Query(default="v1"),
) -> EstimationResponse:
    """Generate an estimation from a session description and optional text attachments."""
    session = Session.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        extracted_attachments = await extract_attachment_texts(attachments)
        normalized_description = _normalize_description(
            description, has_attachments=bool(extracted_attachments)
        )
        request = EstimationRequest(
            description=normalized_description,
            project_type=project_type,
            detail_level=detail_level,
            output_format=output_format,
            reference_projects=_parse_reference_projects(reference_projects),
            attachments=extracted_attachments or None,
            preprocessing=preprocessing,
            use_examples=use_examples,
            num_examples=num_examples,
            example_format=example_format,
            model=model,
            max_tokens=max_tokens,
            evaluate=evaluate,
        )
        messages = session.history.to_messages_list(
            request,
            prompt_version=prompt_version,
            project_metadata=session.metadata,
        )
        response = format_response(
            generate_estimation(
                request,
                prompt_version=prompt_version,
                project_metadata=session.metadata,
                messages=messages,
                use_cache=False,
            ),
            prompt_version=prompt_version,
        )
        if request.evaluate:
            response.validation = evaluate_estimation_structure(
                response.estimation,
                response.finish_reason,
                request.output_format,
            )
        session.metadata = extract_project_metadata(
            previous_metadata=session.metadata,
            request=request,
            llm_response=response.estimation,
        )
        response.project_metadata = session.metadata.model_dump()
    except AttachmentTextExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except UnsupportedAttachmentTypeError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except LLMServiceError as e:
        log.error(
            "session_estimation_endpoint_error", session_id=session_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e)) from e

    session.turn_count += 1
    session.history.add_message("user", request.description)
    session.history.add_message("assistant", response.estimation)
    turn_observation = _build_turn_observation(
        session=session,
        request=request,
        response=response,
    )
    session.last_turn_observation = turn_observation
    log.info("turn_observed", **turn_observation)
    return response


def _normalize_description(description: str, *, has_attachments: bool) -> str:
    normalized_description = description.strip()
    if normalized_description:
        return normalized_description

    if has_attachments:
        return "Estimar el proyecto con base en los adjuntos proporcionados."

    raise HTTPException(
        status_code=400,
        detail="description is required when no attachments are provided.",
    )


def _parse_reference_projects(
    reference_projects: str | None,
) -> list[ReferenceProject] | None:
    if not reference_projects:
        return None

    try:
        projects = TypeAdapter(list[ReferenceProject]).validate_python(
            json.loads(reference_projects)
        )
    except (JSONDecodeError, ValidationError) as e:
        raise HTTPException(
            status_code=400,
            detail="reference_projects must be a valid JSON array of reference projects.",
        ) from e

    return projects or None


def _build_turn_observation(
    *,
    session: Session,
    request: EstimationRequest,
    response: EstimationResponse,
) -> dict[str, object]:
    attachments_text = "\n".join(
        attachment.content for attachment in request.attachments or []
    )
    enriched_transcript = "\n".join(
        part for part in (request.description, attachments_text) if part
    )

    return {
        "turn_index": session.turn_count,
        "session_id": session.session_id,
        "enriched_transcript_chars": len(enriched_transcript),
        "attachments_total_chars": len(attachments_text),
        "messages_in_window": sum(len(turn.messages) for turn in session.history.turns),
        "anchors_count": 0,
        "summary_chars": 0,
        "tokens_in": response.usage.input_tokens or 0,
        "tokens_out": response.usage.output_tokens or 0,
        "cost_usd": response.cost_usd or response.usage.cost_estimate or 0.0,
        "latency_ms": response.latency_ms,
        "cache_hit_kind": "exact" if response.cache_hit else "none",
        "last_resolved_tier": None,
    }
