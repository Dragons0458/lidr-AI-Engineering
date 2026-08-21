"""Session 15 failure-mode fixtures: defect present in the fixture, absent in prod files.

No Docker, no network. ``pyyaml`` is a production dependency.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "exercises" / "session-15" / "failure_modes"
COMPOSE = ROOT / "docker-compose.yml"
DOCKERFILE = ROOT / "Dockerfile"
ENV_EXAMPLE = ROOT / ".env.example"

FIXTURE_NAMES = (
    "01-image-does-not-build.Dockerfile",
    "02-wrong-boot-order.yml",
    "03-localhost-vs-service-name.yml",
    "04-ports-leak.yml",
    "05-token-mismatch.env",
)

PROD_TARGETS = ("builder", "base", "ai-service", "web")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _env_value(text: str, key: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        if name.strip() == key:
            return value.strip()
    return None


def test_all_five_fixtures_are_present() -> None:
    missing = [name for name in FIXTURE_NAMES if not (FIXTURES / name).is_file()]
    assert missing == []


def test_fixture_1_copies_everything_before_uv_sync_real_dockerfile_does_not() -> None:
    fixture = (FIXTURES / "01-image-does-not-build.Dockerfile").read_text(
        encoding="utf-8"
    )
    real = DOCKERFILE.read_text(encoding="utf-8")

    copy_idx = fixture.index("COPY . .")
    sync_idx = fixture.index("uv sync")
    assert copy_idx < sync_idx
    assert "AS builder" not in fixture
    assert "AS ai-service" not in fixture
    assert "AS web" not in fixture

    real_sync = real.index("uv sync")
    real_copy_app = real.index("COPY app/")
    assert real_sync < real_copy_app
    for target in PROD_TARGETS:
        assert f"AS {target}" in real


def test_fixture_2_uses_list_depends_on_real_compose_uses_conditions() -> None:
    fixture = _load_yaml(FIXTURES / "02-wrong-boot-order.yml")
    real = _load_yaml(COMPOSE)

    for name, service in fixture["services"].items():
        depends = service.get("depends_on")
        if depends is None:
            continue
        assert isinstance(depends, list), f"{name} should demonstrate list depends_on"

    migrate = real["services"]["migrate"]["depends_on"]
    assert isinstance(migrate, dict)
    assert migrate["postgres"]["condition"] == "service_healthy"

    ai = real["services"]["ai-service"]["depends_on"]
    assert isinstance(ai, dict)
    assert ai["migrate"]["condition"] == "service_completed_successfully"

    for _name, service in real["services"].items():
        depends = service.get("depends_on")
        if not depends:
            continue
        assert isinstance(depends, dict)
        for dep_name, spec in depends.items():
            assert "condition" in spec
            if spec["condition"] == "service_healthy":
                assert "healthcheck" in real["services"][dep_name]


def test_fixture_3_points_at_localhost_real_web_uses_compose_dns() -> None:
    fixture = _load_yaml(FIXTURES / "03-localhost-vs-service-name.yml")
    real = _load_yaml(COMPOSE)

    fixture_url = fixture["services"]["web"]["environment"]["ESTIMATION_API_BASE_URL"]
    assert "localhost" in fixture_url

    real_url = real["services"]["web"]["environment"]["ESTIMATION_API_BASE_URL"]
    assert real_url == "http://ai-service:8000/api/v1"
    assert "localhost" not in real_url


def test_fixture_4_publishes_ai_service_real_compose_only_publishes_web() -> None:
    fixture = _load_yaml(FIXTURES / "04-ports-leak.yml")
    real = _load_yaml(COMPOSE)

    assert fixture["services"]["ai-service"].get("ports")

    published = {
        name for name, service in real["services"].items() if service.get("ports")
    }
    assert published == {"web"}


def test_fixture_5_mismatches_tokens_env_example_keeps_the_alias() -> None:
    fixture = (FIXTURES / "05-token-mismatch.env").read_text(encoding="utf-8")
    example = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert _env_value(fixture, "AI_SERVICE_TOKEN") != _env_value(
        fixture, "ESTIMATE_API_KEY"
    )

    assert _env_value(example, "AI_SERVICE_TOKEN")
    assert _env_value(example, "ESTIMATE_API_KEY")
    assert _env_value(example, "AI_SERVICE_TOKEN") == _env_value(
        example, "ESTIMATE_API_KEY"
    )
