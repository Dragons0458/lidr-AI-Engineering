from __future__ import annotations

import pytest

from streamlit_ui.agents import (
    PRESET_PROFILES,
    AgentProfile,
    compact_config,
    get_default_hourly_rate_eur,
    normalize_agent_trace,
    phase_payload,
    profile_defaults,
    validate_avatar,
)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"reasoning_effort": "extreme"}, "effort"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 21}, "max_iterations"),
        ({"search_top_k": 0}, "search_top_k"),
        ({"search_top_k": 31}, "search_top_k"),
        ({"search_distance_threshold": -0.01}, "search_distance_threshold"),
        ({"search_distance_threshold": 2.01}, "search_distance_threshold"),
    ],
)
def test_profile_rejects_invalid_config_ranges(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentProfile(name="Invalid", config=config)


def test_profile_validates_name_and_persona() -> None:
    with pytest.raises(ValueError, match="empty"):
        AgentProfile(name=" ")
    with pytest.raises(ValueError, match="120"):
        AgentProfile(name="x" * 121)
    with pytest.raises(ValueError, match="2000"):
        AgentProfile(name="Valid", persona="x" * 2001)
    assert (
        AgentProfile(
            name=" Valid ",
            persona=" persona ",
            config={
                "reasoning_effort": "minimal",
                "max_iterations": 1,
                "search_top_k": 30,
                "search_distance_threshold": 0,
            },
        ).name
        == "Valid"
    )


def test_compact_config_removes_inherited_values_but_keeps_zero() -> None:
    assert compact_config(
        {"model": " ", "reasoning_effort": None, "distance": 0, "enabled": False}
    ) == {"distance": 0, "enabled": False}


def test_phase_payload_only_sends_fields_for_requested_phase() -> None:
    profile = AgentProfile(
        name="Custom",
        persona="Search per task.",
        config={
            "model": "gpt-5",
            "reasoning_effort": "high",
            "max_iterations": 15,
            "search_top_k": 8,
            "search_distance_threshold": 0.4,
            "ignored": "value",
        },
    )

    assert phase_payload(profile, "structure") == {
        "model": "gpt-5",
        "reasoning_effort": "high",
        "persona": "Search per task.",
    }
    assert profile.phase_payload("hours") == {
        "model": "gpt-5",
        "reasoning_effort": "high",
        "max_iterations": 15,
        "search_top_k": 8,
        "search_distance_threshold": 0.4,
        "persona": "Search per task.",
    }
    assert phase_payload(None, "hours") == {}


def test_effective_defaults_and_hourly_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    assert profile_defaults(None) == {
        "model": "gpt-5",
        "reasoning_effort": "medium",
        "max_iterations": 10,
        "search_top_k": 5,
        "search_distance_threshold": 0.45,
    }
    assert (
        profile_defaults(AgentProfile(name="Override", config={"search_top_k": 9}))[
            "search_top_k"
        ]
        == 9
    )
    assert get_default_hourly_rate_eur() == 75.0


def test_presets_match_session_12_contract() -> None:
    presets = {profile.name: profile for profile in PRESET_PROFILES}
    assert set(presets) == {"Estándar", "Veloz (debug)", "Exhaustivo"}
    assert presets["Estándar"].is_default is True
    assert presets["Estándar"].config == {
        "model": "gpt-5",
        "reasoning_effort": "medium",
    }
    assert presets["Veloz (debug)"].config["max_iterations"] == 6
    assert presets["Exhaustivo"].config["max_iterations"] == 15
    assert presets["Exhaustivo"].config["search_top_k"] == 8
    assert all("tarea" in profile.persona.lower() for profile in PRESET_PROFILES)


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (b"\x89PNG\r\n\x1a\npayload", "image/png"),
        (b"\xff\xd8\xff\xe0payload", "image/jpeg"),
        (b"GIF87apayload", "image/gif"),
        (b"GIF89apayload", "image/gif"),
        (b"RIFF\x04\x00\x00\x00WEBPpayload", "image/webp"),
    ],
)
def test_avatar_magic_bytes_are_detected(data: bytes, mime: str) -> None:
    assert validate_avatar(data, mime) == mime


def test_avatar_rejects_fake_mime_bad_format_and_oversize() -> None:
    with pytest.raises(ValueError, match="does not match"):
        validate_avatar(b"\x89PNG\r\n\x1a\npayload", "image/jpeg")
    with pytest.raises(ValueError, match="PNG, JPEG, GIF, or WEBP"):
        validate_avatar(b"<svg></svg>", "image/svg+xml")
    with pytest.raises(ValueError, match="byte limit"):
        validate_avatar(b"\xff\xd8\xff" + b"x" * 10, "image/jpeg", max_bytes=10)


def test_trace_normalization_with_and_without_reasoning_summary() -> None:
    trace = {
        "steps": [
            {
                "step": 1,
                "reasoning_summary": "Search for an analogue.",
                "tool": "search_budgets",
                "tool_args": {"query": "OAuth"},
                "observation": "one result",
            },
            {
                "step": 2,
                "reasoning_summary": None,
                "tool": "derive_task_hours",
            },
        ]
    }
    normalized = normalize_agent_trace(trace)
    assert normalized[0]["reasoning_summary"] == "Search for an analogue."
    assert normalized[1]["reasoning_summary"] == "(sin resumen de razonamiento)"
    assert normalized[1]["tool_args"] == {}
    assert normalize_agent_trace(None) == []
