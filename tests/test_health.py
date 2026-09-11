import pytest
from fastapi.testclient import TestClient

from app import __version__
from app.config import get_settings


def test_health_returns_ok(client: TestClient):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_version_and_providers(client: TestClient):
    body = client.get("/health").json()

    assert body["version"] == __version__
    assert body["embedding_provider"]
    assert body["llm_provider"]


def test_health_never_leaks_credentials(client: TestClient):
    """Guard against someone adding the settings dump to this payload later."""
    body = client.get("/health").json()

    assert not any("key" in field.lower() for field in body)


def test_health_reports_models(client: TestClient):
    body = client.get("/health").json()

    assert body["llm_model"]
    assert body["embedding_model"]
    assert body["embedding_dimensions"] > 0


def test_health_reports_missing_credentials(client: TestClient):
    """No key configured in the test environment, so this must be False --
    it is the fastest way to diagnose a misconfigured deployment."""
    assert client.get("/health").json()["credentials_configured"] is False


def test_health_reports_present_credentials_without_revealing_them(
    monkeypatch: pytest.MonkeyPatch, data_dir, client: TestClient
):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-do-not-leak")
    get_settings.cache_clear()

    body = client.get("/health").json()

    assert body["credentials_configured"] is True
    assert "secret-value-do-not-leak" not in str(body)
