from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_embedding_provider
from app.config import Settings, get_settings
from app.main import create_app
from tests.fakes import FakeEmbeddingProvider

MAX_UPLOAD_MB = 1  # keep the oversized-upload test cheap


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the app at a throwaway data directory for one test."""
    target = tmp_path / "data"
    # Ignore the developer's real .env. Without this the suite passes or fails
    # depending on whose machine it runs on -- a configured API key alone was
    # enough to break a test.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(target))
    monkeypatch.setenv("MAX_UPLOAD_MB", str(MAX_UPLOAD_MB))
    # Settings are cached per process, so the cache has to be dropped for the
    # patched environment to be picked up -- and again afterwards, so a stale
    # tmp_path never leaks into the next test.
    get_settings.cache_clear()
    yield target
    get_settings.cache_clear()


@pytest.fixture
def app(data_dir: Path):
    """The application, exposed so tests can override dependencies on it."""
    return create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    """A client whose app has completed startup.

    Used as a context manager on purpose: that is what runs FastAPI's lifespan
    hook, which creates the directories and applies migrations.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def embeddings(app) -> FakeEmbeddingProvider:
    """Substitute a deterministic provider. No test may reach the network:
    it would be slow, need a key, and fail on someone else's rate limit."""
    provider = FakeEmbeddingProvider()
    app.dependency_overrides[get_embedding_provider] = lambda: provider
    return provider


@pytest.fixture
def uploads_dir(data_dir: Path) -> Path:
    return data_dir / "uploads"
