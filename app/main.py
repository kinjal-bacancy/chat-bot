"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app import __version__
from app.api.health import router as health_router
from app.config import get_settings


def create_app() -> FastAPI:
    """Application factory.

    Building the app inside a function (rather than at import time) keeps
    tests able to construct an isolated instance against different settings.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        debug=settings.debug,
    )
    app.include_router(health_router)
    return app


app = create_app()
