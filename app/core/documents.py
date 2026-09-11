"""Document records: the metadata half of an upload.

`status` tracks a document's position in the ingestion pipeline. Every stage
built in later steps advances it, which is what makes a stalled or failed
document visible instead of silently missing from search results.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.core.storage import StoredFile

DocumentStatus = Literal["uploaded", "parsed", "chunked", "indexed", "failed"]


@dataclass(frozen=True)
class Document:
    id: str
    filename: str
    content_type: str
    extension: str
    size_bytes: int
    sha256: str
    stored_path: Path
    status: DocumentStatus
    error: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Document":
        return cls(
            id=row["id"],
            filename=row["filename"],
            content_type=row["content_type"],
            extension=row["extension"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            stored_path=Path(row["stored_path"]),
            status=row["status"],
            error=row["error"],
            created_at=row["created_at"],
        )


def create(conn: sqlite3.Connection, stored: StoredFile, filename: str) -> Document:
    conn.execute(
        """
        INSERT INTO documents (
            id, filename, content_type, extension, size_bytes,
            sha256, stored_path, status, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'uploaded', NULL, ?)
        """,
        (
            stored.id,
            filename,
            stored.content_type,
            stored.extension,
            stored.size_bytes,
            stored.sha256,
            str(stored.path),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    found = get(conn, stored.id)
    assert found is not None  # just inserted
    return found


def get(conn: sqlite3.Connection, document_id: str) -> Document | None:
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?", (document_id,)
    ).fetchone()
    return Document.from_row(row) if row else None


def find_by_sha256(conn: sqlite3.Connection, sha256: str) -> Document | None:
    row = conn.execute(
        "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
    ).fetchone()
    return Document.from_row(row) if row else None


def list_all(conn: sqlite3.Connection, *, limit: int = 100) -> list[Document]:
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY created_at DESC, id LIMIT ?", (limit,)
    ).fetchall()
    return [Document.from_row(row) for row in rows]


def set_status(
    conn: sqlite3.Connection,
    document_id: str,
    status: DocumentStatus,
    *,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE documents SET status = ?, error = ? WHERE id = ?",
        (status, error, document_id),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, document_id: str) -> bool:
    """Remove a document's row and its file on disk.

    The row goes first: an orphaned file wastes disk, whereas a row pointing
    at a file that no longer exists breaks every later pipeline stage.
    """
    document = get(conn, document_id)
    if document is None:
        return False

    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
    document.stored_path.unlink(missing_ok=True)
    return True
