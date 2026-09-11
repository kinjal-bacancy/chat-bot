"""Hybrid retrieval: dense vectors and keyword search, fused.

The two halves fail in opposite directions. Embeddings capture meaning but
average a passage into one point, so a rare exact term -- a gem name, a
version string -- contributes almost nothing. Keyword search matches those
exactly but cannot connect "stale" to "deprecated by its own maintainers".

Their scores are not comparable, so they are fused by rank rather than by
value. Reciprocal Rank Fusion asks only "how high did each retriever put
this?", which sidesteps the fact that a cosine similarity of 0.75 and a BM25
score of -8.2 mean nothing to each other.
"""

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.core import vectors
from app.providers.base import EmbeddingProvider

SearchMode = Literal["hybrid", "dense", "keyword"]

# Standard RRF constant. Large enough that the top few ranks are close
# together, so one retriever being confident does not overrule the other.
_RRF_K = 60

_WORD = re.compile(r"[A-Za-z0-9_.~-]+")

# Question scaffolding. These appear in nearly every query and nearly every
# chunk, so leaving them in means BM25 ranks on them and buries the one rare
# term that actually distinguishes the answer.
_STOPWORDS = frozenset("""
a an the and or but if is are was were be been being do does did doing have
has had having i we you they it he she this that these those there here what
which who whom whose when where why how all any both each few more most other
some such no nor not only own same so than too very can will just should now
of to in on at by for with about against between into through during before
after above below from up down out off over under again further then once me
my our your their its as
""".split())


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
    # Per-retriever detail, so a result can be explained rather than trusted.
    dense_score: float | None = None
    dense_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None


def to_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    A raw query goes straight into FTS5's own query syntax, where a stray
    quote or a bare '-' is a syntax error rather than a search. Each word is
    extracted and quoted as a literal, then OR-ed: partial matches should
    still rank, since requiring every term finds nothing on real questions.
    """
    words = [w for w in _WORD.findall(query) if w.lower() not in _STOPWORDS]
    if not words:
        # A query made entirely of stopwords has no lexical signal at all;
        # better to contribute nothing than to rank on "what" and "is".
        return ""
    return " OR ".join(f'"{word}"' for word in words)


def _document_filter(document_ids: list[str] | None) -> tuple[str, list[str]]:
    if not document_ids:
        return "", []
    placeholders = ",".join("?" * len(document_ids))
    return placeholders, list(document_ids)


def dense_candidates(
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    query: str,
    limit: int,
    document_ids: list[str] | None,
) -> tuple[list[tuple[str, float]], int]:
    """(chunk_id, cosine) best first, and how many chunks were compared."""
    sql = """
        SELECT c.id, e.vector FROM chunks c
        JOIN chunk_embeddings e
          ON e.sha256 = c.sha256 AND e.model = ? AND e.dimensions = ?
    """
    params: list[object] = [provider.model, provider.dimensions]
    placeholders, ids = _document_filter(document_ids)
    if placeholders:
        sql += f" WHERE c.document_id IN ({placeholders})"
        params.extend(ids)

    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return [], 0

    matrix = vectors.stack([row["vector"] for row in rows])
    scores = vectors.similarities(matrix, provider.embed_query(query))

    count = min(limit, len(rows))
    top = np.argpartition(-scores, count - 1)[:count]
    ordered = top[np.argsort(-scores[top])]
    return [(rows[i]["id"], float(scores[i])) for i in ordered], len(rows)


def keyword_candidates(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    document_ids: list[str] | None,
) -> list[tuple[str, float]]:
    """(chunk_id, relevance) best first, using SQLite's BM25."""
    match = to_fts_query(query)
    if not match:
        return []

    sql = "SELECT chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts WHERE chunks_fts MATCH ?"
    params: list[object] = [match]
    placeholders, ids = _document_filter(document_ids)
    if placeholders:
        sql += f" AND document_id IN ({placeholders})"
        params.extend(ids)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        # A query that still confuses FTS5 should degrade to dense-only
        # rather than fail the whole search.
        return []

    # SQLite returns bm25 as a negative number, better matches more negative.
    # Flipped so that larger is better everywhere in this module.
    return [(row["chunk_id"], -float(row["rank"])) for row in rows]


def _reciprocal_rank_fusion(
    weighted_rankings: list[tuple[list[str], float]],
) -> dict[str, float]:
    """Weighted RRF.

    Weights are not decoration. Keyword search always returns *something* --
    a question like "what version of rails are we on" has no rare terms, so
    BM25 ranks on "version" and "rails" and is confidently wrong. Unweighted
    fusion lets that outvote a correct dense ranking. Dense therefore leads,
    with keyword able to lift a chunk it alone found but not to overturn the
    ordering on its own.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking, weight in weighted_rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += weight / (_RRF_K + position)
    return fused


def _hydrate(
    conn: sqlite3.Connection, chunk_ids: list[str]
) -> dict[str, sqlite3.Row]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, c.page_number, c.chunk_index,
               c.heading, c.text, d.filename
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    return {row["id"]: row for row in rows}


def search(
    conn: sqlite3.Connection,
    provider: EmbeddingProvider,
    query: str,
    *,
    top_k: int = 5,
    mode: SearchMode = "hybrid",
    candidates: int = 50,
    dense_weight: float = 2.0,
    keyword_weight: float = 1.0,
    document_ids: list[str] | None = None,
    min_score: float = -1.0,
) -> tuple[list[SearchHit], int]:
    """Return the best-matching chunks and how many were searched.

    `mode` exists so the two retrievers can be compared directly on the same
    query. Being able to see that keyword search found a chunk dense search
    missed is what makes a retrieval problem diagnosable.
    """
    dense: list[tuple[str, float]] = []
    keyword: list[tuple[str, float]] = []
    searched = 0

    if mode in ("hybrid", "dense"):
        dense, searched = dense_candidates(
            conn, provider, query, candidates, document_ids
        )
    if mode in ("hybrid", "keyword"):
        keyword = keyword_candidates(conn, query, candidates, document_ids)
        if mode == "keyword":
            searched = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    dense_rank = {chunk_id: i for i, (chunk_id, _) in enumerate(dense, start=1)}
    keyword_rank = {chunk_id: i for i, (chunk_id, _) in enumerate(keyword, start=1)}
    dense_score = dict(dense)
    keyword_score = dict(keyword)

    if mode == "hybrid":
        fused = _reciprocal_rank_fusion(
            [(list(dense_rank), dense_weight), (list(keyword_rank), keyword_weight)]
        )
    elif mode == "dense":
        fused = dense_score
    else:
        fused = keyword_score

    ordered = sorted(fused.items(), key=lambda item: -item[1])[:top_k]
    rows = _hydrate(conn, [chunk_id for chunk_id, _ in ordered])

    hits = []
    for chunk_id, score in ordered:
        row = rows.get(chunk_id)
        if row is None or score < min_score:
            continue
        hits.append(
            SearchHit(
                chunk_id=row["id"],
                document_id=row["document_id"],
                filename=row["filename"],
                page_number=row["page_number"],
                chunk_index=row["chunk_index"],
                heading=row["heading"],
                text=row["text"],
                score=float(score),
                dense_score=dense_score.get(chunk_id),
                dense_rank=dense_rank.get(chunk_id),
                keyword_score=keyword_score.get(chunk_id),
                keyword_rank=keyword_rank.get(chunk_id),
            )
        )
    return hits, searched
