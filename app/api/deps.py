"""Shared FastAPI dependencies."""

import sqlite3
from typing import Iterator

from fastapi import Depends

from app.config import Settings, get_settings
from app.core import database


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
