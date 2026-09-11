"""Answering questions from the indexed documents."""

import json
import sqlite3
from typing import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_db, get_embedding_provider, get_llm_provider
from app.config import Settings, get_settings
from app.core import answering, retrieval
from app.core.retrieval import SearchMode
from app.providers import EmbeddingProvider, LLMProvider, ProviderError

router = APIRouter(tags=["ask"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: SearchMode | None = None
    document_ids: list[str] | None = None


class SourceResponse(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    heading: str | None
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    # True when the model found nothing in the sources to stand on. A correct
    # outcome, not an error.
    refused: bool
    # Only the sources the answer actually cited.
    sources: list[SourceResponse]
    # Everything retrieval offered, cited or not -- this is what makes a poor
    # answer diagnosable.
    retrieved: list[SourceResponse]
    searched_chunks: int
    mode: str
    model: str


@router.post("/ask", response_model=AskResponse, summary="Ask a grounded question")
def ask(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    embedder: EmbeddingProvider = Depends(get_embedding_provider),
    llm: LLMProvider = Depends(get_llm_provider),
    conn: sqlite3.Connection = Depends(get_db),
) -> AskResponse:
    """Retrieve, then answer using only what was retrieved.

    The response carries both the sources cited and everything retrieved, so
    a disappointing answer can be attributed without re-running anything: if
    the right passage is absent from `retrieved`, retrieval failed; if it is
    there but uncited, the model did.
    """
    mode = request.mode or settings.search_mode
    hits, searched = retrieval.search(
        conn,
        embedder,
        request.question,
        top_k=request.top_k or settings.search_top_k,
        mode=mode,
        candidates=settings.search_candidates,
        dense_weight=settings.search_dense_weight,
        keyword_weight=settings.search_keyword_weight,
        document_ids=request.document_ids,
    )

    sources = answering.build_sources(hits)

    if not sources:
        # Nothing retrieved: answer without calling the model at all. Sending
        # an empty source list would invite it to answer from its own
        # knowledge, which is the one thing this system must not do.
        return AskResponse(
            question=request.question,
            answer=(
                "No indexed documents matched this question. Upload a document "
                "and POST /documents/{id}/ingest, then ask again."
                if searched == 0
                else "The documents do not contain anything relevant to this question."
            ),
            refused=True,
            sources=[],
            retrieved=[],
            searched_chunks=searched,
            mode=mode,
            model=llm.model,
        )

    try:
        raw = llm.generate(
            answering.SYSTEM_PROMPT,
            answering.build_prompt(request.question, sources),
        )
    except ProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    answer = answering.assemble(raw, sources, llm.model)

    return AskResponse(
        question=request.question,
        answer=answer.text,
        refused=answer.refused,
        sources=[SourceResponse(**vars(s)) for s in answer.sources],
        retrieved=[SourceResponse(**vars(s)) for s in answer.retrieved],
        searched_chunks=searched,
        mode=mode,
        model=answer.model,
    )


def _retrieve(request, settings, embedder, conn):
    """Shared retrieval for both the plain and streaming endpoints."""
    mode = request.mode or settings.search_mode
    hits, searched = retrieval.search(
        conn,
        embedder,
        request.question,
        top_k=request.top_k or settings.search_top_k,
        mode=mode,
        candidates=settings.search_candidates,
        dense_weight=settings.search_dense_weight,
        keyword_weight=settings.search_keyword_weight,
        document_ids=request.document_ids,
    )
    return mode, hits, searched


def _event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


@router.post(
    "/ask/stream",
    summary="Ask a grounded question, streaming the answer",
    response_class=StreamingResponse,
)
def ask_stream(
    request: AskRequest,
    settings: Settings = Depends(get_settings),
    embedder: EmbeddingProvider = Depends(get_embedding_provider),
    llm: LLMProvider = Depends(get_llm_provider),
    conn: sqlite3.Connection = Depends(get_db),
) -> StreamingResponse:
    """Server-sent events: `retrieved`, then `delta` per piece, then `done`.

    Retrieval runs to completion before the stream opens, for two reasons.
    The database connection is a request-scoped dependency and must not be
    read from inside the generator, and sending the retrieved chunks first
    lets a UI fill in its sources panel while the answer is still arriving.
    """
    mode, hits, searched = _retrieve(request, settings, embedder, conn)
    sources = answering.build_sources(hits)
    prompt = answering.build_prompt(request.question, sources) if sources else ""

    def events() -> Iterator[str]:
        yield _event(
            "retrieved",
            {
                "retrieved": [vars(s) for s in sources],
                "searched_chunks": searched,
                "mode": mode,
                "model": llm.model,
            },
        )

        if not sources:
            yield _event(
                "done",
                {
                    "answer": (
                        "No indexed documents matched this question."
                        if searched == 0
                        else "The documents do not contain anything relevant."
                    ),
                    "refused": True,
                    "sources": [],
                },
            )
            return

        collected: list[str] = []
        try:
            for piece in llm.stream(answering.SYSTEM_PROMPT, prompt):
                collected.append(piece)
                yield _event("delta", {"text": piece})
        except ProviderError as exc:
            # The response has already started, so the status code is spent.
            # An error event is the only way left to tell the client.
            yield _event("error", {"detail": str(exc)})
            return

        answer = answering.assemble("".join(collected), sources, llm.model)
        yield _event(
            "done",
            {
                "answer": answer.text,
                "refused": answer.refused,
                "sources": [vars(s) for s in answer.sources],
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
