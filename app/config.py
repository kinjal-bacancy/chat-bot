"""Typed application settings, loaded once from the environment / .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every knob the app has. Field names map to env vars case-insensitively,
    so `data_dir` is populated from DATA_DIR."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated vars already in the shell
    )

    app_name: str = "RAG Chatbot"
    debug: bool = False
    data_dir: Path = Path("./data")

    # Which provider implementation to use. Validated properly in the step
    # that introduces the provider registry.
    embedding_provider: str = "local"
    llm_provider: str = "anthropic"

    # Only the key for the selected provider needs to be set.
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process, and so tests can
    clear the cache to swap in a different environment."""
    return Settings()
