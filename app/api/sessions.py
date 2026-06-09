import json
from json import JSONDecodeError
from typing import Annotated, Literal
from uuid import uuid4

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import TypeAdapter, ValidationError

from app.config import get_settings
from app.foundation.llm.errors import LLMServiceError
from app.foundation.guardrails.input import InputGuardrailViolation
from app.foundation.formatters import format_response
from app.domain.schemas.acb import ACBResponse
from app.domain.schemas.estimation import (
    DetailLevel,
    ExampleFormat,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    PreprocessingMode,
    ProjectType,
    ReferenceProject,
)
from app.domain.schemas.sessions import (
    SessionCreateRequest,
    SessionDebugResponse,
    SessionResponse,
)
from app.foundation.attachments.extractor import (
    AttachmentTextExtractionError,
    UnsupportedAttachmentTypeError,
    extract_attachment_texts,
)
from app.generation.conversation.compression import apply_compression
from app.domain.estimation_service import generate_estimation, generate_estimation_acb
from app.generation.conversation.evaluation import evaluate_estimation_structure
from app.generation.conversation.metadata_extractor import extract_project_metadata
from app.generation.conversation.store import Session
from app.generation.conversation.tier_resolver import Tier

settings = get_settings()
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
        message_count=session.history.recent_message_count(),
        anchors_count=session.history.anchors_count,
        summary_chars=session.history.summary_chars,
        summary=session.history.summary or "",
        anchors=[anchor.content for anchor in session.history.anchors],
        metadata=session.metadata.model_dump(),
        last_resolved_tier=session.last_resolved_tier,
        last_tier_rule=session.last_tier_rule,
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
    tier: Annotated[str | None, Form()] = None,
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
        _apply_tier_resolution(session, request, tier_override=tier)
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
        if request.evaluate and not response.out_of_scope:
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except InputGuardrailViolation as e:
        raise HTTPException(
            status_code=400,
            detail={"reason": e.reason, "message": e.message},
        ) from e
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
    _apply_history_policy(session)
    turn_observation = _build_turn_observation(
        session=session,
        request=request,
        response=response,
    )
    session.last_turn_observation = turn_observation
    log.info("turn_observed", **turn_observation)
    return response


@router.post("/sessions/{session_id}/estimate-acb", response_model=ACBResponse)
async def estimate_session_acb(
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
    attachments: Annotated[list[UploadFile] | None, File()] = None,
    tier: Annotated[str | None, Form()] = None,
    prompt_version: Literal["v3"] = Query(default="v3"),
) -> ACBResponse:
    """Structured estimation with Actor-Critic-Boss."""
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
            evaluate=False,
        )
        acb_response = generate_estimation_acb(
            request,
            session=session,
            tier_override=tier,
            prompt_version=prompt_version,
        )
        session.metadata = extract_project_metadata(
            previous_metadata=session.metadata,
            request=request,
            llm_response=acb_response.result.summary,
        )
        acb_response.project_metadata = session.metadata.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except InputGuardrailViolation as e:
        raise HTTPException(
            status_code=400,
            detail={"reason": e.reason, "message": e.message},
        ) from e
    except AttachmentTextExtractionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except UnsupportedAttachmentTypeError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except LLMServiceError as e:
        log.error("session_acb_endpoint_error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    session.turn_count += 1
    session.history.add_message("user", request.description)
    session.history.add_message("assistant", acb_response.result.model_dump_json())
    _apply_history_policy(session)
    session.last_turn_observation = _build_acb_turn_observation(
        session=session,
        request=request,
        response=acb_response,
    )
    log.info("turn_observed", **session.last_turn_observation)
    return acb_response


def _enriched_transcript(request: EstimationRequest) -> str:
    attachments_text = "\n".join(
        attachment.content for attachment in request.attachments or []
    )
    return "\n".join(part for part in (request.description, attachments_text) if part)


def _apply_tier_resolution(
    session: Session,
    request: EstimationRequest,
    *,
    tier_override: str | None,
) -> None:
    if not settings.TIER_RESOLUTION_ENABLED and not tier_override:
        return

    override: Tier | None = None
    if tier_override:
        try:
            override = Tier(tier_override)
        except ValueError as exc:
            raise ValueError(f"Invalid tier: {tier_override}") from exc

    from app.generation.conversation.tier_resolver import resolve_tier

    resolved, rule = resolve_tier(
        transcript=_enriched_transcript(request),
        metadata=session.metadata,
        override=override,
    )
    session.last_resolved_tier = resolved.value
    session.last_tier_rule = rule


def _apply_history_policy(session: Session) -> None:
    if settings.MEMORY_COMPRESSION_ENABLED:
        from app.dependencies import get_llm_wrapper, get_runtime_config

        runtime = get_runtime_config()
        apply_compression(
            session.history,
            llm_wrapper=get_llm_wrapper(),
            compression_model=runtime.effective("COMPRESSION_MODEL"),
            anchor_detection_mode=settings.ANCHOR_DETECTION_MODE,
        )
    else:
        session.history.trim_to_max_turns()


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
    enriched_transcript = _enriched_transcript(request)
    attachments_text = "\n".join(
        attachment.content for attachment in request.attachments or []
    )

    return {
        "turn_index": session.turn_count,
        "session_id": session.session_id,
        "enriched_transcript_chars": len(enriched_transcript),
        "attachments_total_chars": len(attachments_text),
        "messages_in_window": session.history.recent_message_count(),
        "anchors_count": session.history.anchors_count,
        "summary_chars": session.history.summary_chars,
        "tokens_in": response.usage.input_tokens or 0,
        "tokens_out": response.usage.output_tokens or 0,
        "cost_usd": response.cost_usd or response.usage.cost_estimate or 0.0,
        "latency_ms": response.latency_ms,
        "cache_hit_kind": "exact" if response.cache_hit else "none",
        "last_resolved_tier": session.last_resolved_tier,
        "last_tier_rule": session.last_tier_rule,
    }


def _build_acb_turn_observation(
    *,
    session: Session,
    request: EstimationRequest,
    response: ACBResponse,
) -> dict[str, object]:
    enriched_transcript = _enriched_transcript(request)
    attachments_text = "\n".join(
        attachment.content for attachment in request.attachments or []
    )
    return {
        "turn_index": session.turn_count,
        "session_id": session.session_id,
        "enriched_transcript_chars": len(enriched_transcript),
        "attachments_total_chars": len(attachments_text),
        "messages_in_window": session.history.recent_message_count(),
        "anchors_count": session.history.anchors_count,
        "summary_chars": session.history.summary_chars,
        "tokens_in": response.usage.input_tokens or 0,
        "tokens_out": response.usage.output_tokens or 0,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "cache_hit_kind": "none",
        "last_resolved_tier": session.last_resolved_tier,
        "last_tier_rule": session.last_tier_rule,
        "acb_final_decision": response.acb.final_decision,
        "acb_iterations_run": response.acb.iterations_run,
    }
