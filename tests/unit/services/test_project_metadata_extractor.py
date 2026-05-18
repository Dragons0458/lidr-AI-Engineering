from app.schemas.estimation import (
    DetailLevel,
    EstimationRequest,
    OutputFormat,
    ProjectType,
)
from app.services.project_metadata_extractor import extract_project_metadata
from app.services.sessions import ProjectMetadata


def test_extract_project_metadata_uses_latest_metadata_state(
    monkeypatch,
) -> None:
    captured = {}

    def fake_create_project_metadata_completion(messages):
        captured["messages"] = messages
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
    request = EstimationRequest(
        description="Proyecto Portal Clientes con React y FastAPI.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )
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
    def fake_create_project_metadata_completion(messages):
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
    request = EstimationRequest(
        description="Ya no queremos React ni Firebase; usar FastAPI.",
        project_type=ProjectType.WEB_SAAS,
        detail_level=DetailLevel.MEDIUM,
        output_format=OutputFormat.LINE_ITEMS,
    )

    metadata = extract_project_metadata(
        previous_metadata=ProjectMetadata(
            mentioned_technologies=["React", "Firebase", "FastAPI"]
        ),
        request=request,
        llm_response="Estimacion usando FastAPI.",
    )

    assert metadata.mentioned_technologies == ["FastAPI"]
    assert metadata.excluded_technologies == ["React", "Firebase"]
