"""Tests for get_async_openai_client dependency."""

from __future__ import annotations

from unittest.mock import patch

from app.dependencies import get_async_openai_client, get_openai_client


def test_get_async_openai_client_returns_client_when_key_set():
    get_async_openai_client.cache_clear()
    get_openai_client.cache_clear()
    with patch("app.dependencies.get_settings") as mock_settings:
        mock_settings.return_value.OPENAI_API_KEY = "sk-test"
        client = get_async_openai_client()
        assert client is not None
    get_async_openai_client.cache_clear()


def test_get_async_openai_client_returns_none_without_key():
    get_async_openai_client.cache_clear()
    with patch("app.dependencies.get_settings") as mock_settings:
        mock_settings.return_value.OPENAI_API_KEY = None
        assert get_async_openai_client() is None
    get_async_openai_client.cache_clear()
