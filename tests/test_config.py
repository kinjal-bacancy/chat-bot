import pytest
from pydantic import ValidationError

from app.config import Settings


def test_defaults_select_gemini():
    settings = Settings()

    assert settings.embedding_provider == "gemini"
    assert settings.llm_provider == "gemini"


def test_unimplemented_provider_is_rejected():
    """A typo or an aspirational value fails immediately, rather than at the
    first embedding call halfway through an ingestion run."""
    with pytest.raises(ValidationError):
        Settings(embedding_provider="openai")

    with pytest.raises(ValidationError):
        Settings(llm_provider="anthropic")


def test_credentials_configured_reflects_the_key():
    assert Settings(gemini_api_key=None).credentials_configured is False
    assert Settings(gemini_api_key="xyz").credentials_configured is True
