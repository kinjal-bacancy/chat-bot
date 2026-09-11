"""Shared FastAPI dependencies."""

import sqlite3
from typing import Iterator

from fastapi import Depends

from app.config import Settings, get_settings
from app.core import database
from app.providers import (
    EmbeddingProvider,
    LLMProvider,
    build_embedding_provider,
    build_llm_provider,
)


def get_db(settings: Settings = Depends(get_settings)) -> Iterator[sqlite3.Connection]:
    """One SQLite connection per request, closed when the response is done.

    Yielding (rather than returning) is what lets FastAPI run the cleanup
    after the handler finishes, including when it raised.
    """
    conn = database.connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


def get_embedding_provider(
    settings: Settings = Depends(get_settings),
) -> EmbeddingProvider:
    """The configured embedding backend.

    A dependency rather than a module-level singleton so tests can substitute
    a fake and never reach the network.
    """
    return build_embedding_provider(settings)


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    return build_llm_provider(settings)
