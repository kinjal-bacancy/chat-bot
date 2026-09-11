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
    embedding_model: str
    embedding_dimensions: int
    llm_provider: str
    llm_model: str
    search_mode: str
    credentials_configured: bool


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report that the API is up, and which provider backends it is wired to.

    Echoing the provider and model selection makes a whole class of "why is it
    behaving like that" question answerable with one request. Reports whether
    a credential is present, never the credential itself.
    """
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        search_mode=settings.search_mode,
        credentials_configured=settings.credentials_configured,
    )
