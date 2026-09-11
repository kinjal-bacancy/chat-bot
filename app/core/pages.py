"""Extracted page text.

Kept in the database rather than re-parsed on demand: parsing is the slowest
stage, and chunking will be re-run repeatedly while its strategy is tuned.
"""

import sqlite3

from app.core.parsers import Page, ParsedDocument


def replace_for_document(
    conn: sqlite3.Connection, document_id: str, parsed: ParsedDocument
) -> None:
    """Store a document's pages, discarding any from a previous parse.

    Replacing rather than appending keeps re-parsing idempotent, which
    matters because a parser fix should be applied by simply parsing again.
    """
    with conn:  # one transaction: never leave a half-replaced document
        conn.execute("DELETE FROM pages WHERE document_id = ?", (document_id,))
        conn.executemany(
            "INSERT INTO pages (document_id, page_number, text) VALUES (?, ?, ?)",
            [(document_id, page.number, page.text) for page in parsed.pages],
        )


def list_for_document(conn: sqlite3.Connection, document_id: str) -> list[Page]:
    rows = conn.execute(
        "SELECT page_number, text FROM pages WHERE document_id = ? ORDER BY page_number",
        (document_id,),
    ).fetchall()
    return [Page(number=row["page_number"], text=row["text"]) for row in rows]


def count_for_document(conn: sqlite3.Connection, document_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM pages WHERE document_id = ?", (document_id,)
    ).fetchone()[0]
