"""Retrieval over indexed chunks."""

import sqlite3

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_db, get_embedding_provider
from app.config import Settings, get_settings
from app.core import retrieval
from app.providers import EmbeddingProvider

router = APIRouter(tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    # Restrict to particular documents; omit to search everything indexed.
    document_ids: list[str] | None = None
    # Defaults to filtering nothing; raise it to suppress weak matches.
    min_score: float = Field(default=-1.0, ge=-1.0, le=1.0)


class SearchHitResponse(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    heading: str | None
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHitResponse]
    # How many chunks were actually compared. Zero here means nothing is
    # indexed yet, which otherwise looks identical to "nothing matched".
    searched_chunks: int
    model: str
    # Set only when the search could not have worked, so an empty result is
    # never mistaken for "nothing in your documents matched".
    note: str | None = None


@router.post("/search", response_model=SearchResponse, summary="Search indexed chunks")
def search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    conn: sqlite3.Connection = Depends(get_db),
) -> SearchResponse:
    """Return the chunks most similar to a query, with their scores.

    The scores are the point of this endpoint. Reading them is how you tell a
    retrieval problem from a generation problem later: a wrong answer built on
    chunks that scored 0.8 is the model's fault, and one built on chunks that
    scored 0.3 is retrieval's.
    """
    hits, searched = retrieval.search(
        conn,
        provider,
        request.query,
        top_k=request.top_k or settings.search_top_k,
        document_ids=request.document_ids,
        min_score=request.min_score,
    )

    return SearchResponse(
        query=request.query,
        hits=[SearchHitResponse(**vars(hit)) for hit in hits],
        searched_chunks=searched,
        model=provider.model,
        note=(
            "No chunks are indexed for this model. Upload a document, then "
            "POST /documents/{id}/ingest."
            if searched == 0
            else None
        ),
    )
