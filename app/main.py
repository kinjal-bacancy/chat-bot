"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.config import Settings, get_settings
from app.core import database


def _prepare_storage(settings: Settings) -> None:
    """Create the data directories and bring the schema up to date."""
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    conn = database.connect(settings.db_path)
    try:
        database.migrate(conn)
    finally:
        conn.close()


def create_app() -> FastAPI:
    """Application factory.

    Building the app inside a function (rather than at import time) keeps
    tests able to construct an isolated instance against different settings.
    """
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Runs once on startup, before the first request is served, so no
        # handler can encounter a missing directory or an un-migrated schema.
        _prepare_storage(settings)
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(ask_router)
    return app


app = create_app()
