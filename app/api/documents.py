"""Document upload and listing."""

import sqlite3

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel

from app.api.deps import get_db, get_embedding_provider
from app.config import Settings, get_settings
from app.core import chunks as chunks_repo
from app.core import pipeline
from app.core import documents as documents_repo
from app.core import pages as pages_repo
from app.core import parsers, storage
from app.core.documents import Document
from app.providers import EmbeddingProvider, ProviderError

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    error: str | None
    created_at: str
    # True when these bytes were already on file, so nothing new was stored.
    duplicate: bool = False

    @classmethod
    def of(cls, document: Document, *, duplicate: bool = False) -> "DocumentResponse":
        # stored_path is deliberately not exposed: server paths are not the
        # client's business, and the ID is enough to address a document.
        return cls(
            id=document.id,
            filename=document.filename,
            content_type=document.content_type,
            size_bytes=document.size_bytes,
            sha256=document.sha256,
            status=document.status,
            error=document.error,
            created_at=document.created_at,
            duplicate=duplicate,
        )


@router.post("", response_model=DocumentResponse, summary="Upload a document")
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> DocumentResponse:
    """Store an uploaded document and record it as `uploaded`.

    Re-uploading identical bytes returns the existing record rather than
    creating a second copy. Uploading the same document twice is a normal
    accident, and duplicated chunks would pollute retrieval with near-identical
    results that crowd out genuinely distinct sources.
    """
    try:
        document, duplicate = await pipeline.store_upload(
            conn, settings, file, file.filename
        )
    except storage.UnsupportedFileType as exc:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{exc.extension}'. "
                f"Allowed: {', '.join(sorted(storage.ALLOWED_EXTENSIONS))}"
            ),
        ) from exc
    except storage.UploadTooLarge as exc:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit",
        ) from exc
    except storage.EmptyUpload as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        ) from exc

    return DocumentResponse.of(document, duplicate=duplicate)


@router.get("", response_model=list[DocumentResponse], summary="List documents")
def list_documents(
    limit: int = 100,
    conn: sqlite3.Connection = Depends(get_db),
) -> list[DocumentResponse]:
    return [DocumentResponse.of(d) for d in documents_repo.list_all(conn, limit=limit)]


@router.get("/{document_id}", response_model=DocumentResponse, summary="Get a document")
def get_document(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> DocumentResponse:
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentResponse.of(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document",
)
def delete_document(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> Response:
    if not documents_repo.delete(conn, document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class PageResponse(BaseModel):
    number: int
    text: str


class ParseResponse(BaseModel):
    id: str
    status: str
    page_count: int
    character_count: int


@router.post(
    "/{document_id}/parse",
    response_model=ParseResponse,
    summary="Extract text from a document",
)
def parse_document(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> ParseResponse:
    """Extract text and store it per page.

    Safe to call repeatedly: a re-parse replaces the previous extraction, so
    fixing a parser is a matter of running this again rather than re-uploading.
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        result = pipeline.parse(conn, document)
    except parsers.ParserError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    return ParseResponse(
        id=document_id,
        status="parsed",
        page_count=result.page_count,
        character_count=result.character_count,
    )


@router.get(
    "/{document_id}/pages",
    response_model=list[PageResponse],
    summary="Read the extracted text",
)
def get_document_pages(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> list[PageResponse]:
    """Return the text exactly as the chunker will see it.

    Worth actually reading. Extraction damage -- merged columns, tables
    flattened into nonsense, missing sections -- is invisible downstream and
    surfaces much later as answers that are subtly wrong.

    An unparsed document is an error rather than an empty list, because an
    empty list would be indistinguishable from a document that parsed
    successfully and genuinely contained nothing.
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    if document.status == "uploaded":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"Document has not been parsed yet. "
                f"POST /documents/{document_id}/parse first."
            ),
        )

    if document.status == "failed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"Parsing failed for this document: {document.error}",
        )

    return [
        PageResponse(number=page.number, text=page.text)
        for page in pages_repo.list_for_document(conn, document_id)
    ]


class ChunkResponse(BaseModel):
    index: int
    page_number: int
    heading: str | None
    text: str
    characters: int
    char_start: int
    char_end: int


class ChunkSummary(BaseModel):
    id: str
    status: str
    chunk_count: int
    total_characters: int
    average_characters: int
    settings: dict[str, int]


@router.post(
    "/{document_id}/chunk",
    response_model=ChunkSummary,
    summary="Split a parsed document into chunks",
)
def chunk_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    conn: sqlite3.Connection = Depends(get_db),
) -> ChunkSummary:
    """Chunk a document's extracted text.

    Safe to re-run: chunks are replaced, not appended, because tuning the
    chunk size is an iterative business.
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        pipeline.chunk(conn, settings, document_id)
    except pipeline.StageNotReady as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    stored = chunks_repo.list_for_document(conn, document_id)
    total = sum(len(chunk.text) for chunk in stored)
    return ChunkSummary(
        id=document_id,
        status="chunked",
        chunk_count=len(stored),
        total_characters=total,
        average_characters=round(total / len(stored)) if stored else 0,
        settings={
            "target_chars": settings.chunk_target_chars,
            "overlap_chars": settings.chunk_overlap_chars,
            "max_chars": settings.chunk_max_chars,
        },
    )


@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
    summary="Read the chunks",
)
def get_document_chunks(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> list[ChunkResponse]:
    """Return chunks exactly as they will be embedded.

    The point of this endpoint is to be read. Chunk boundaries are the single
    biggest influence on retrieval quality, and a bad boundary is obvious on
    sight but invisible in an aggregate metric.
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    stored = chunks_repo.list_for_document(conn, document_id)
    if not stored:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=(
                f"Document has not been chunked yet. "
                f"POST /documents/{document_id}/chunk first."
            ),
        )

    return [
        ChunkResponse(
            index=chunk.chunk_index,
            page_number=chunk.page_number,
            heading=chunk.heading,
            text=chunk.text,
            characters=len(chunk.text),
            char_start=chunk.char_start,
            char_end=chunk.char_end,
        )
        for chunk in stored
    ]


class EmbedResponse(BaseModel):
    id: str
    status: str
    chunk_count: int
    embedded: int
    reused: int
    model: str
    dimensions: int


@router.post(
    "/{document_id}/embed",
    response_model=EmbedResponse,
    summary="Embed a document's chunks",
)
def embed_document(
    document_id: str,
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    conn: sqlite3.Connection = Depends(get_db),
) -> EmbedResponse:
    """Ensure every chunk has a vector, reusing cached ones where the text is
    unchanged.

    Re-running after a chunking change costs API calls only for text that
    actually differs, which is what makes tuning chunk size affordable.
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        run = pipeline.embed(conn, provider, document_id)
    except pipeline.StageNotReady as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return EmbedResponse(
        id=document_id,
        status="indexed",
        chunk_count=run.chunk_count,
        embedded=run.embedded,
        reused=run.reused,
        model=run.model,
        dimensions=run.dimensions,
    )


class IngestResponse(BaseModel):
    id: str
    status: str
    page_count: int
    character_count: int
    chunk_count: int
    embedded: int
    reused: int
    model: str
    dimensions: int


@router.post(
    "/{document_id}/ingest",
    response_model=IngestResponse,
    summary="Parse, chunk and embed in one call",
)
def ingest_document(
    document_id: str,
    settings: Settings = Depends(get_settings),
    provider: EmbeddingProvider = Depends(get_embedding_provider),
    conn: sqlite3.Connection = Depends(get_db),
) -> IngestResponse:
    """Run the whole pipeline.

    The individual stage endpoints exist for debugging -- each fails for
    different reasons and is worth inspecting alone. This is the normal path:
    a document that is parsed but not embedded is silently unsearchable, and
    nothing about a successful /parse response suggests two more calls are
    outstanding.

    Safe to re-run. Each stage replaces its own output and embedding reuses
    cached vectors, so re-ingesting an unchanged document costs no API calls.
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    try:
        result = pipeline.ingest(conn, settings, provider, document)
    except parsers.ParserError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except ProviderError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    fields = vars(result) | {"id": result.document_id}
    fields.pop("document_id")
    return IngestResponse(**fields)
