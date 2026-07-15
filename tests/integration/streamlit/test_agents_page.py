from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from streamlit_ui.store import (
    delete_agent_profile,
    get_agent_profile,
    get_default_agent_profile,
    list_agent_profiles,
)

ROOT = Path(__file__).resolve().parents[3]
PAGE = ROOT / "streamlit_ui/pages/7_Agentes.py"
PNG = b"\x89PNG\r\n\x1a\navatar"
WEBP = b"RIFF\x04\x00\x00\x00WEBPavatar"


@pytest.fixture
def agents_app(tmp_path, monkeypatch) -> tuple[AppTest, str]:
    db_path = str(tmp_path / "agents-page.db")
    monkeypatch.setenv("STREAMLIT_DB_PATH", db_path)
    for profile in list_agent_profiles(db_path=db_path):
        delete_agent_profile(profile["id"], db_path=db_path)
    monkeypatch.setattr(
        "streamlit_ui.common.fetch_available_agent_models",
        lambda *args, **kwargs: ["gpt-5", "gpt-5-mini"],
    )
    return AppTest.from_file(str(PAGE)).run(timeout=10), db_path


def _save_new_profile(
    app: AppTest,
    *,
    name: str,
    is_default: bool = False,
    avatar: tuple[str, bytes, str] | None = None,
) -> AppTest:
    app.text_input[0].input(name)
    app.selectbox[1].select("gpt-5")
    app.text_area[0].input("Search historical analogues independently per task.")
    if is_default:
        app.checkbox[-1].check()
    if avatar:
        app.file_uploader[0].upload(*avatar)
    return app.button[0].click().run(timeout=10)


def test_create_edit_change_default_and_delete_profile(agents_app) -> None:
    app, db_path = agents_app
    app = _save_new_profile(app, name="Primary", is_default=True)
    primary = list_agent_profiles(db_path=db_path)[0]
    assert primary["name"] == "Primary"
    assert primary["is_default"] is True

    app.selectbox[0].select(None).run()
    app = _save_new_profile(app, name="Secondary")
    profiles = {
        profile["name"]: profile for profile in list_agent_profiles(db_path=db_path)
    }
    secondary_id = profiles["Secondary"]["id"]

    app.selectbox[0].select(secondary_id).run()
    app.text_input[0].input("Secondary edited")
    app.button[0].click().run()
    assert (
        get_agent_profile(secondary_id, db_path=db_path)["name"] == "Secondary edited"
    )

    app.selectbox[0].select(secondary_id).run()
    next(
        button for button in app.button if button.label == "Marcar como default"
    ).click().run()
    assert get_default_agent_profile(db_path=db_path)["id"] == secondary_id

    app.selectbox[0].select(secondary_id).run()
    next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Confirmo que quiero eliminarlo"
    ).check()
    next(
        button for button in app.button if button.label == "Eliminar perfil"
    ).click().run()
    assert get_agent_profile(secondary_id, db_path=db_path) is None


def test_upload_replace_and_remove_avatar(agents_app) -> None:
    app, db_path = agents_app
    app = _save_new_profile(
        app,
        name="Avatar profile",
        avatar=("avatar.png", PNG, "image/png"),
    )
    profile = list_agent_profiles(db_path=db_path)[0]
    profile_id = profile["id"]
    assert profile["avatar_bytes"] == PNG

    app.selectbox[0].select(profile_id).run()
    app.file_uploader[0].upload("avatar.webp", WEBP, "image/webp")
    app.button[0].click().run()
    assert get_agent_profile(profile_id, db_path=db_path)["avatar_bytes"] == WEBP

    app.selectbox[0].select(profile_id).run()
    next(
        checkbox
        for checkbox in app.checkbox
        if checkbox.label == "Eliminar avatar actual"
    ).check()
    app.button[0].click().run()
    loaded = get_agent_profile(profile_id, db_path=db_path)
    assert loaded["avatar_bytes"] is None
    assert loaded["avatar_content_type"] is None


def test_page_shows_profile_and_avatar_validation_errors(agents_app) -> None:
    app, db_path = agents_app
    app.button[0].click().run()
    assert any("empty" in error.value for error in app.error)
    assert list_agent_profiles(db_path=db_path) == []

    app.text_input[0].input("Fake avatar")
    app.file_uploader[0].upload("fake.png", b"<svg/>", "image/png")
    app.button[0].click().run()
    assert any("PNG, JPEG, GIF, or WEBP" in error.value for error in app.error)
    assert list_agent_profiles(db_path=db_path) == []
