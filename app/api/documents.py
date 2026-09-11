"""Document upload and listing."""

import sqlite3

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from pydantic import BaseModel

from app.api.deps import get_db
from app.config import Settings, get_settings
from app.core import documents as documents_repo
from app.core import pages as pages_repo
from app.core import parsers, storage
from app.core.documents import Document

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
        stored = await storage.save_upload(
            file,
            file.filename,
            uploads_dir=settings.uploads_dir,
            max_bytes=settings.max_upload_bytes,
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

    existing = documents_repo.find_by_sha256(conn, stored.sha256)
    if existing is not None:
        stored.path.unlink(missing_ok=True)  # drop the redundant copy
        return DocumentResponse.of(existing, duplicate=True)

    document = documents_repo.create(conn, stored, filename=file.filename or "untitled")
    return DocumentResponse.of(document)


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
        parsed = parsers.parse(document.stored_path, document.extension)
    except parsers.ParserError as exc:
        # Recorded on the document so a failure is visible in the listing
        # rather than only in this one response.
        documents_repo.set_status(conn, document_id, "failed", error=str(exc))
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    pages_repo.replace_for_document(conn, document_id, parsed)
    documents_repo.set_status(conn, document_id, "parsed")

    return ParseResponse(
        id=document_id,
        status="parsed",
        page_count=len(parsed.pages),
        character_count=parsed.character_count,
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
    """
    document = documents_repo.get(conn, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    return [
        PageResponse(number=page.number, text=page.text)
        for page in pages_repo.list_for_document(conn, document_id)
    ]
