"""Typed application settings, loaded once from the environment / .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Providers that actually have an implementation. Adding one means adding it
# here too, so an unimplemented value fails at startup with a clear error
# rather than at the first embedding call.
EmbeddingProviderName = Literal["gemini"]
LLMProviderName = Literal["gemini"]


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

    # Reject uploads larger than this. Enforced while streaming, so an
    # oversized file is never held in memory or fully written to disk.
    max_upload_mb: int = 25

    # Chunking, in characters. Characters rather than tokens because Gemini's
    # tokeniser is server-side, and roughly four characters per token is close
    # enough for sizing. Changing any of these requires a re-index.
    chunk_target_chars: int = 1200   # ~300 tokens
    chunk_overlap_chars: int = 200   # carried as whole blocks, never a fragment
    chunk_max_chars: int = 2000      # a single block above this is force-split

    # Retrieval
    search_top_k: int = 5
    # How many candidates each retriever contributes before fusion. Wider
    # than top_k on purpose: a chunk ranked 20th by one retriever and 2nd by
    # the other should still surface.
    search_candidates: int = 50
    # Measured on the gem audit: dense 0.933 MRR, hybrid 0.883, keyword
    # 0.803 over 10 questions. Hybrid is the usual recommendation and is
    # kept ready, but defaulting to it here would be following advice over
    # evidence. Revisit against the evaluation set on a larger corpus.
    search_mode: str = "dense"    # hybrid | dense | keyword
    # Relative pull of each retriever during fusion. Dense leads because
    # keyword search answers every query, including ones with no rare terms
    # in them, and an unweighted fusion lets it outvote a correct ranking.
    search_dense_weight: float = 2.0
    search_keyword_weight: float = 1.0

    # Which provider implementation to use.
    embedding_provider: EmbeddingProviderName = "gemini"
    llm_provider: LLMProviderName = "gemini"

    # Named GEMINI_API_KEY because the google-genai SDK reads that variable
    # itself, so its client needs no explicit wiring.
    gemini_api_key: str | None = None

    llm_model: str = "gemini-3.8-flash"
    # Zero by default: the task is to restate the sources faithfully, and
    # sampling variety in that job reads as invention.
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 1024

    embedding_model: str = "gemini-embedding-001"
    # gemini-embedding-001 returns 3072 dimensions by default but is trained
    # so that a truncated prefix is still a usable embedding. 768 is Google's
    # recommended size and makes the index four times smaller for a small
    # quality cost. Changing this invalidates every stored vector.
    embedding_dimensions: int = 768
    # Texts per embedding request. Larger means fewer round trips but a
    # coarser retry unit when the free tier rate-limits a batch.
    embedding_batch_size: int = 32

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def credentials_configured(self) -> bool:
        """Whether a key is present for the selected providers. Never exposes
        the key itself -- only whether one was supplied."""
        return bool(self.gemini_api_key)


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is parsed once per process, and so tests can
    clear the cache to swap in a different environment."""
    return Settings()
