"""Liveness and configuration introspection."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    embedding_provider: str
    llm_provider: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report that the API is up, and which provider backends it is wired to.

    Echoing the provider selection makes a whole class of "why is it behaving
    like that" question answerable with one request. Deliberately reports the
    provider *names* and never the credentials.
    """
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
    )
