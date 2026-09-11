"""Provider selection.

Which implementation is used is a configuration decision, made here and
nowhere else.
"""

from app.config import Settings
from app.providers.base import (
    EmbeddingProvider,
    LLMProvider,
    MissingCredentials,
    ProviderError,
)
from app.providers.gemini import GeminiEmbeddingProvider, GeminiLLMProvider

__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "MissingCredentials",
    "ProviderError",
    "build_embedding_provider",
    "build_llm_provider",
]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "gemini":
        if not settings.gemini_api_key:
            raise MissingCredentials(
                "GEMINI_API_KEY is not set. Add it to .env -- get a key at "
                "https://aistudio.google.com/apikey"
            )
        return GeminiEmbeddingProvider(
            api_key=settings.gemini_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        )

    # Unreachable while the setting is a Literal, but an explicit failure
    # beats an implicit None if that ever changes.
    raise ProviderError(f"Unknown embedding provider: {settings.embedding_provider}")


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise MissingCredentials(
                "GEMINI_API_KEY is not set. Add it to .env -- get a key at "
                "https://aistudio.google.com/apikey"
            )
        return GeminiLLMProvider(
            api_key=settings.gemini_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            thinking_budget=settings.llm_thinking_budget,
        )

    raise ProviderError(f"Unknown LLM provider: {settings.llm_provider}")
