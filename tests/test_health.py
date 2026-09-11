from fastapi.testclient import TestClient

from app import __version__


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
