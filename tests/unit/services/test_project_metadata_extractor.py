from app.config import get_settings
from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.services.project_metadata_extractor import (
    _create_project_metadata_completion,
    extract_project_metadata,
)
from app.services.sessions import ProjectMetadata


def _request(**overrides) -> EstimationRequest:
    values = {
        "description": "Proyecto Portal Clientes con React y FastAPI.",
        "project_type": ProjectType.WEB_SAAS,
        "detail_level": DetailLevel.MEDIUM,
        "output_format": OutputFormat.LINE_ITEMS,
    }
    values.update(overrides)
    return EstimationRequest(**values)


def test_extract_project_metadata_uses_latest_metadata_state(
    monkeypatch,
) -> None:
    captured = {}

    def fake_create_project_metadata_completion(messages, *, request):
        captured["messages"] = messages
        captured["model"] = request.model
        return ProjectMetadata(
            project_name="Portal Clientes",
            assumed_team_size=4,
            mentioned_technologies=["React", "FastAPI"],
            agreed_scope="Autenticacion, pagos y reportes.",
        )

    monkeypatch.setattr(
        "app.services.project_metadata_extractor._create_project_metadata_completion",
        fake_create_project_metadata_completion,
    )
    request = _request()
    previous_metadata = ProjectMetadata(mentioned_technologies=["PostgreSQL", "react"])

    metadata = extract_project_metadata(
        previous_metadata=previous_metadata,
        request=request,
        llm_response="Estimacion para autenticacion, pagos y reportes.",
    )

    assert metadata.project_name == "Portal Clientes"
    assert metadata.assumed_team_size == 4
    assert metadata.mentioned_technologies == ["React", "FastAPI"]
    assert metadata.agreed_scope == "Autenticacion, pagos y reportes."
    assert "<previous_project_metadata>" in captured["messages"][1]["content"]
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"


def test_extract_project_metadata_removes_excluded_technologies(monkeypatch) -> None:
    def fake_create_project_metadata_completion(messages, *, request):
        return ProjectMetadata(
            project_name="Portal Clientes",
            mentioned_technologies=["FastAPI", "React", "Firebase"],
            excluded_technologies=["React", "Firebase"],
            agreed_scope="Usar FastAPI y evitar React/Firebase.",
        )

    monkeypatch.setattr(
        "app.services.project_metadata_extractor._create_project_metadata_completion",
        fake_create_project_metadata_completion,
    )

    metadata = extract_project_metadata(
        previous_metadata=ProjectMetadata(
            mentioned_technologies=["React", "Firebase", "FastAPI"]
        ),
        request=_request(description="Ya no queremos React ni Firebase; usar FastAPI."),
        llm_response="Estimacion usando FastAPI.",
    )

    assert metadata.mentioned_technologies == ["FastAPI"]
    assert metadata.excluded_technologies == ["React", "Firebase"]


def test_create_project_metadata_completion_uses_primary_model(monkeypatch) -> None:
    captured = {}

    def fake_instructor_create(**kwargs):
        captured.update(kwargs)
        return ProjectMetadata(project_name="Test")

    monkeypatch.setattr(
        "app.services.project_metadata_extractor._client.chat.completions.create",
        fake_instructor_create,
    )
    get_settings.cache_clear()

    _create_project_metadata_completion(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ],
        request=_request(),
    )

    assert captured["model"] == get_settings().PRIMARY_MODEL
    assert captured["max_retries"] == get_settings().LLM_RETRIES
    assert captured["timeout"] == get_settings().LLM_TIMEOUT


def test_create_project_metadata_completion_honors_request_model_override(
    monkeypatch,
) -> None:
    captured = {}

    def fake_instructor_create(**kwargs):
        captured.update(kwargs)
        return ProjectMetadata(project_name="Test")

    monkeypatch.setattr(
        "app.services.project_metadata_extractor._client.chat.completions.create",
        fake_instructor_create,
    )

    _create_project_metadata_completion(
        [{"role": "user", "content": "user"}],
        request=_request(model="gemini/gemini-2.5-flash"),
    )

    assert captured["model"] == "gemini/gemini-2.5-flash"


def test_instructor_client_uses_llm_wrapper_completion() -> None:
    from app.services.llm_wrapper import completion as wrapper_completion

    import app.services.project_metadata_extractor as extractor

    assert extractor._client is not None
    assert wrapper_completion.__name__ == "completion"
    assert wrapper_completion.__module__ == "app.services.llm_wrapper"
