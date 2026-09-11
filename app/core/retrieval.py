"""Searching the indexed chunks.

Deliberately built and exposed three steps before any LLM call. When an
answer is wrong, the first question is whether the right passage was even
retrieved, and that has to be answerable without a model in the way.
"""

import sqlite3
from dataclasses import dataclass

import numpy as np

from app.core import vectors
from app.providers.base import EmbeddingProvider


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    heading: str | None
    text: str
    score: float


def _candidate_rows(
    conn: sqlite3.Connection,
    model: str,
    dimensions: int,
    document_ids: list[str] | None,
) -> list[sqlite3.Row]:
    """Chunks that have a vector for this model and size.

    An inner join, so a chunk whose embedding is missing or was made with a
    different model is simply not searchable -- better than comparing against
    a vector from another space and returning confident nonsense.
    """
    sql = """
        SELECT c.id, c.document_id, c.page_number, c.chunk_index,
               c.heading, c.text, e.vector, d.filename
        FROM chunks c
        JOIN chunk_embeddings e
          ON e.sha256 = c.sha256 AND e.model = ? AND e.dimensions = ?
        JOIN documents d ON d.id = c.document_id
    """
    params: list[object] = [model, dimensions]

    if document_ids:
        placeholders = ",".join("?" * len(document_ids))
        sql += f" WHERE c.document_id IN ({placeholders})"
        params.extend(document_ids)

    return conn.execute(sql, params).fetchall()


def search(
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    query: str,
    *,
    top_k: int = 5,
    document_ids: list[str] | None = None,
    # -1.0 filters nothing: cosine similarity is bounded below by -1, and a
    # default floor would silently drop hits rather than let them rank.
    min_score: float = -1.0,
) -> tuple[list[SearchHit], int]:
    """Return the best-matching chunks and how many were searched."""
    rows = _candidate_rows(conn, provider.model, provider.dimensions, document_ids)
    if not rows:
        return [], 0

    matrix = vectors.stack([row["vector"] for row in rows])
    scores = vectors.similarities(matrix, provider.embed_query(query))

    # argpartition finds the top k without sorting all of them; only the k
    # survivors are then ordered.
    count = min(top_k, len(rows))
    top = np.argpartition(-scores, count - 1)[:count]
    ordered = top[np.argsort(-scores[top])]

    hits = [
        SearchHit(
            chunk_id=rows[index]["id"],
            document_id=rows[index]["document_id"],
            filename=rows[index]["filename"],
            page_number=rows[index]["page_number"],
            chunk_index=rows[index]["chunk_index"],
            heading=rows[index]["heading"],
            text=rows[index]["text"],
            score=float(scores[index]),
        )
        for index in ordered
        if float(scores[index]) >= min_score
    ]
    return hits, len(rows)
