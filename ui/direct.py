"""In-process backend, for hosts that run a single process.

Streamlit Community Cloud runs one process, so there is no uvicorn to talk
to. This calls `app.core` directly and presents the same surface as the HTTP
client, so the UI does not know or care which it is using.

That this is possible at all is the payoff of `app/core` importing no web
framework: the pipeline was never coupled to being served over HTTP.
"""

import asyncio
import io
import sqlite3
from dataclasses import asdict
from typing import Iterator

from app import __version__
from app.config import Settings, get_settings
from app.core import answering, chunks as chunks_repo, database
from app.core import documents as documents_repo
from app.core import parsers, pipeline, retrieval, storage
from app.providers import (
    ProviderError,
    build_embedding_provider,
    build_llm_provider,
)

from api import ApiError


class _BytesReader:
    """The slice of an upload that `storage.save_upload` expects."""

    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)

    async def read(self, size: int) -> bytes:
        return self._buffer.read(size)


class DirectBackend:
    """Same methods as `ui.api.Api`, without the network."""

    base_url = "in-process"

    def __init__(self) -> None:
        self._settings: Settings = get_settings()
        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            database.migrate(conn)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return database.connect(self._settings.db_path)

    def _embedder(self):
        try:
            return build_embedding_provider(self._settings)
        except ProviderError as exc:
            raise ApiError(str(exc)) from exc

    # --- documents ----------------------------------------------------

    def health(self) -> dict:
        s = self._settings
        return {
            "status": "ok",
            "app_name": s.app_name,
            "version": __version__,
            "embedding_provider": s.embedding_provider,
            "embedding_model": s.embedding_model,
            "embedding_dimensions": s.embedding_dimensions,
            "llm_provider": s.llm_provider,
            "llm_model": s.llm_model,
            "search_mode": s.search_mode,
            "credentials_configured": s.credentials_configured,
        }

    def documents(self) -> list[dict]:
        conn = self._connect()
        try:
            return [
                {**asdict(d), "stored_path": None, "duplicate": False}
                for d in documents_repo.list_all(conn)
            ]
        finally:
            conn.close()

    def upload(self, name: str, data: bytes) -> dict:
        conn = self._connect()
        try:
            document, duplicate = asyncio.run(
                pipeline.store_upload(conn, self._settings, _BytesReader(data), name)
            )
        except storage.UnsupportedFileType as exc:
            allowed = ", ".join(sorted(storage.ALLOWED_EXTENSIONS))
            raise ApiError(f"{exc}. Allowed: {allowed}") from exc
        except (storage.UploadTooLarge, storage.EmptyUpload) as exc:
            raise ApiError(str(exc)) from exc
        finally:
            conn.close()
        return {**asdict(document), "stored_path": None, "duplicate": duplicate}

    def ingest(self, document_id: str) -> dict:
        conn = self._connect()
        try:
            document = documents_repo.get(conn, document_id)
            if document is None:
                raise ApiError("Document not found")
            result = pipeline.ingest(
                conn, self._settings, self._embedder(), document
            )
            return asdict(result)
        except parsers.ParserError as exc:
            raise ApiError(str(exc)) from exc
        except ProviderError as exc:
            raise ApiError(str(exc)) from exc
        finally:
            conn.close()

    def delete(self, document_id: str) -> None:
        conn = self._connect()
        try:
            if not documents_repo.delete(conn, document_id):
                raise ApiError("Document not found")
        finally:
            conn.close()

    def chunks(self, document_id: str) -> list[dict]:
        conn = self._connect()
        try:
            return [asdict(c) for c in chunks_repo.list_for_document(conn, document_id)]
        finally:
            conn.close()

    # --- asking -------------------------------------------------------

    def _retrieve(self, question: str, body: dict):
        s = self._settings
        conn = self._connect()
        try:
            return retrieval.search(
                conn,
                self._embedder(),
                question,
                top_k=body.get("top_k") or s.search_top_k,
                mode=body.get("mode") or s.search_mode,
                candidates=s.search_candidates,
                dense_weight=s.search_dense_weight,
                keyword_weight=s.search_keyword_weight,
                document_ids=body.get("document_ids"),
            )
        finally:
            conn.close()

    def search(self, query: str, **body) -> dict:
        hits, searched = self._retrieve(query, body)
        return {
            "query": query,
            "hits": [asdict(h) for h in hits],
            "searched_chunks": searched,
            "model": self._settings.embedding_model,
            "mode": body.get("mode") or self._settings.search_mode,
            "note": None if searched else "Nothing is indexed yet. Upload a document.",
        }

    def ask_stream(self, question: str, **body) -> Iterator[tuple[str, dict]]:
        """Yield the same events the HTTP endpoint sends."""
        mode = body.get("mode") or self._settings.search_mode
        hits, searched = self._retrieve(question, body)
        sources = answering.build_sources(hits)

        try:
            llm = build_llm_provider(self._settings)
        except ProviderError as exc:
            yield "error", {"detail": str(exc)}
            return

        yield "retrieved", {
            "retrieved": [asdict(s) for s in sources],
            "searched_chunks": searched,
            "mode": mode,
            "model": llm.model,
        }

        if not sources:
            # Never call the model with no sources: it would answer from its
            # own knowledge, which is the one thing this system must not do.
            yield "done", {
                "answer": (
                    "No indexed documents matched this question."
                    if searched == 0
                    else "The documents do not contain anything relevant."
                ),
                "refused": True,
                "sources": [],
            }
            return

        collected: list[str] = []
        try:
            for piece in llm.stream(
                answering.SYSTEM_PROMPT, answering.build_prompt(question, sources)
            ):
                collected.append(piece)
                yield "delta", {"text": piece}
        except ProviderError as exc:
            yield "error", {"detail": str(exc)}
            return

        answer = answering.assemble("".join(collected), sources, llm.model)
        yield "done", {
            "answer": answer.text,
            "refused": answer.refused,
            "sources": [asdict(s) for s in answer.sources],
        }
