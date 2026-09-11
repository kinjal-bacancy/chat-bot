"""Embedding chunks, with a cache that survives re-chunking.

The cache is keyed by the hash of a chunk's text, not by chunk id. Re-running
the chunker produces new chunk rows, but most of their text is byte-identical
to what was embedded before, so only genuinely new text costs an API call.

Without this, the thing that stops you tuning chunk size is the rate limit --
and chunk size is the parameter most worth tuning.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core import chunks as chunks_repo
from app.core import vectors
from app.providers.base import EmbeddingProvider


@dataclass(frozen=True)
class EmbeddingRun:
    chunk_count: int
    embedded: int   # sent to the provider
    reused: int     # served from cache
    model: str
    dimensions: int


def cached_hashes(
    conn: sqlite3.Connection, hashes: list[str], model: str, dimensions: int
) -> set[str]:
    if not hashes:
        return set()

    found: set[str] = set()
    # Chunked to stay under SQLite's variable limit on a large document.
    for start in range(0, len(hashes), 500):
        window = hashes[start : start + 500]
        placeholders = ",".join("?" * len(window))
        rows = conn.execute(
            f"""
            SELECT sha256 FROM chunk_embeddings
            WHERE model = ? AND dimensions = ? AND sha256 IN ({placeholders})
            """,
            (model, dimensions, *window),
        ).fetchall()
        found.update(row["sha256"] for row in rows)
    return found


def store(
    conn: sqlite3.Connection,
    pairs: list[tuple[str, list[float]]],
    model: str,
    dimensions: int,
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunk_embeddings
                (sha256, model, dimensions, vector, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (sha256, model, dimensions, vectors.to_blob(vector), now)
                for sha256, vector in pairs
            ],
        )


def embed_document(
    conn: sqlite3.Connection, provider: EmbeddingProvider, document_id: str
) -> EmbeddingRun:
    """Ensure every chunk of a document has a stored vector."""
    stored_chunks = chunks_repo.list_for_document(conn, document_id)

    # Deduplicate by hash: a repeated block -- a boilerplate paragraph, an
    # overlap tail -- is one vector, not several identical ones.
    by_hash = {chunk.sha256: chunk.text for chunk in stored_chunks}

    already = cached_hashes(conn, list(by_hash), provider.model, provider.dimensions)
    missing = {h: text for h, text in by_hash.items() if h not in already}

    if missing:
        hashes = list(missing)
        produced = provider.embed_documents([missing[h] for h in hashes])
        store(conn, list(zip(hashes, produced)), provider.model, provider.dimensions)

    return EmbeddingRun(
        chunk_count=len(stored_chunks),
        embedded=len(missing),
        reused=len(by_hash) - len(missing),
        model=provider.model,
        dimensions=provider.dimensions,
    )


def coverage(
    conn: sqlite3.Connection, document_id: str, model: str, dimensions: int
) -> tuple[int, int]:
    """(chunks with a vector, total chunks) for this model and size."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN e.sha256 IS NULL THEN 0 ELSE 1 END) AS embedded
        FROM chunks c
        LEFT JOIN chunk_embeddings e
               ON e.sha256 = c.sha256 AND e.model = ? AND e.dimensions = ?
        WHERE c.document_id = ?
        """,
        (model, dimensions, document_id),
    ).fetchone()
    return (row["embedded"] or 0), (row["total"] or 0)
