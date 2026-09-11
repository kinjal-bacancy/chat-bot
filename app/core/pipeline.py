"""The ingestion pipeline: parse -> chunk -> embed.

Each stage is separately callable, because each is separately debuggable and
they fail for different reasons. But running them one at a time is a
debugging affordance, not a workflow: the normal path is all three, and
having to remember the order is how a document ends up silently unsearchable.
"""

import sqlite3
from dataclasses import dataclass

from app.config import Settings
from app.core import chunking
from app.core import chunks as chunks_repo
from app.core import documents as documents_repo
from app.core import embeddings as embeddings_repo
from app.core import pages as pages_repo
from app.core import parsers, storage
from app.core.documents import Document
from app.core.storage import _AsyncReadable
from app.providers.base import EmbeddingProvider, ProviderError


class StageNotReady(Exception):
    """A stage was asked to run before the one it depends on."""


@dataclass(frozen=True)
class ParseResult:
    page_count: int
    character_count: int


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    status: str
    page_count: int
    character_count: int
    chunk_count: int
    embedded: int
    reused: int
    model: str
    dimensions: int


async def store_upload(
    conn: sqlite3.Connection,
    settings: Settings,
    source: _AsyncReadable,
    filename: str | None,
) -> tuple[Document, bool]:
    """Save an upload and record it. Returns the document and whether these
    bytes were already on file.

    Re-uploading identical bytes returns the existing record rather than
    creating a second copy. Duplicate documents would seed near-identical
    chunks that compete for the same top-k slots and crowd out genuinely
    distinct sources.
    """
    stored = await storage.save_upload(
        source,
        filename,
        uploads_dir=settings.uploads_dir,
        max_bytes=settings.max_upload_bytes,
    )

    existing = documents_repo.find_by_sha256(conn, stored.sha256)
    if existing is not None:
        stored.path.unlink(missing_ok=True)  # drop the redundant copy
        return existing, True

    return documents_repo.create(conn, stored, filename=filename or "untitled"), False


def parse(conn: sqlite3.Connection, document: Document) -> ParseResult:
    """Extract text. Records the failure on the document before re-raising,
    so a broken document is visible in the listing and not just in whichever
    response happened to trigger it."""
    try:
        parsed = parsers.parse(document.stored_path, document.extension)
    except parsers.ParserError as exc:
        documents_repo.set_status(conn, document.id, "failed", error=str(exc))
        raise

    pages_repo.replace_for_document(conn, document.id, parsed)
    documents_repo.set_status(conn, document.id, "parsed")
    return ParseResult(len(parsed.pages), parsed.character_count)


def chunk(conn: sqlite3.Connection, settings: Settings, document_id: str) -> int:
    stored_pages = pages_repo.list_for_document(conn, document_id)
    if not stored_pages:
        raise StageNotReady(
            f"Document has no extracted text. "
            f"POST /documents/{document_id}/parse first."
        )

    produced = chunking.chunk_pages(
        [(page.number, page.text) for page in stored_pages],
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
        max_chars=settings.chunk_max_chars,
    )
    chunks_repo.replace_for_document(conn, document_id, produced)
    documents_repo.set_status(conn, document_id, "chunked")
    return len(produced)


def embed(
    conn: sqlite3.Connection, provider: EmbeddingProvider, document_id: str
) -> embeddings_repo.EmbeddingRun:
    if chunks_repo.count_for_document(conn, document_id) == 0:
        raise StageNotReady(
            f"Document has no chunks. POST /documents/{document_id}/chunk first."
        )

    try:
        run = embeddings_repo.embed_document(conn, provider, document_id)
    except ProviderError as exc:
        documents_repo.set_status(conn, document_id, "failed", error=str(exc))
        raise

    documents_repo.set_status(conn, document_id, "indexed")
    return run


def ingest(
    conn: sqlite3.Connection,
    settings: Settings,
    provider: EmbeddingProvider,
    document: Document,
) -> IngestResult:
    """Run every stage. Safe to re-run: each stage replaces its own output,
    and embedding reuses cached vectors for text that has not changed."""
    parsed = parse(conn, document)
    chunk_count = chunk(conn, settings, document.id)
    run = embed(conn, provider, document.id)

    return IngestResult(
        document_id=document.id,
        status="indexed",
        page_count=parsed.page_count,
        character_count=parsed.character_count,
        chunk_count=chunk_count,
        embedded=run.embedded,
        reused=run.reused,
        model=run.model,
        dimensions=run.dimensions,
    )
