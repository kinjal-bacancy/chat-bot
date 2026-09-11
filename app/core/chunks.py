"""Stored chunks."""

import sqlite3
from dataclasses import dataclass

from app.core.chunking import Chunk


@dataclass(frozen=True)
class StoredChunk:
    id: str
    document_id: str
    page_number: int
    chunk_index: int
    heading: str | None
    text: str
    char_start: int
    char_end: int
    sha256: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "StoredChunk":
        return cls(**{key: row[key] for key in row.keys()})


def replace_for_document(
    conn: sqlite3.Connection, document_id: str, chunks: list[Chunk]
) -> None:
    """Store a document's chunks, discarding any previous ones.

    Re-chunking is expected to happen repeatedly while the strategy is tuned,
    so this is a replace rather than an append.
    """
    with conn:
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.executemany(
            """
            INSERT INTO chunks (
                id, document_id, page_number, chunk_index,
                heading, text, char_start, char_end, sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    chunk.new_id(),
                    document_id,
                    chunk.page_number,
                    chunk.index,
                    chunk.heading,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.sha256,
                )
                for chunk in chunks
            ],
        )


def list_for_document(conn: sqlite3.Connection, document_id: str) -> list[StoredChunk]:
    rows = conn.execute(
        "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index", (document_id,)
    ).fetchall()
    return [StoredChunk.from_row(row) for row in rows]


def count_for_document(conn: sqlite3.Connection, document_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
    ).fetchone()[0]
