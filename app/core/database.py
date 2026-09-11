"""SQLite access and schema migrations.

Migrations are an ordered list of DDL scripts. SQLite tracks how many have
been applied in `PRAGMA user_version`, so startup applies only what's new.
Crude compared to Alembic, but the schema here is small and this keeps the
whole data model readable in one file.
"""

import sqlite3
from pathlib import Path

# Append-only. Never edit a script that has shipped -- add a new one.
MIGRATIONS: list[str] = [
    # 1: documents
    """
    CREATE TABLE documents (
        id            TEXT    PRIMARY KEY,
        filename      TEXT    NOT NULL,   -- as uploaded; for display only
        content_type  TEXT    NOT NULL,
        extension     TEXT    NOT NULL,
        size_bytes    INTEGER NOT NULL,
        sha256        TEXT    NOT NULL UNIQUE,
        stored_path   TEXT    NOT NULL,
        status        TEXT    NOT NULL,   -- uploaded|parsed|chunked|indexed|failed
        error         TEXT,
        created_at    TEXT    NOT NULL
    );
    CREATE INDEX idx_documents_created_at ON documents (created_at DESC);
    """,
    # 2: extracted text, one row per page
    """
    CREATE TABLE pages (
        document_id  TEXT    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        page_number  INTEGER NOT NULL,
        text         TEXT    NOT NULL,
        PRIMARY KEY (document_id, page_number)
    );
    """,
    # 3: chunks
    """
    CREATE TABLE chunks (
        id           TEXT    PRIMARY KEY,
        document_id  TEXT    NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
        page_number  INTEGER NOT NULL,
        chunk_index  INTEGER NOT NULL,
        heading      TEXT,
        text         TEXT    NOT NULL,
        char_start   INTEGER NOT NULL,
        char_end     INTEGER NOT NULL,
        sha256       TEXT    NOT NULL,
        UNIQUE (document_id, chunk_index)
    );
    CREATE INDEX idx_chunks_document ON chunks (document_id, chunk_index);
    CREATE INDEX idx_chunks_sha256 ON chunks (sha256);
    """,
    # 4: cached chunk embeddings, keyed by the text's hash rather than by
    #    chunk id, so re-chunking reuses vectors for text that did not change
    """
    CREATE TABLE chunk_embeddings (
        sha256      TEXT    NOT NULL,
        model       TEXT    NOT NULL,
        dimensions  INTEGER NOT NULL,
        vector      BLOB    NOT NULL,
        created_at  TEXT    NOT NULL,
        PRIMARY KEY (sha256, model, dimensions)
    );
    """,
]


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with the pragmas this app assumes."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because FastAPI runs sync dependencies in a
    # worker thread while async handlers run on the event loop, so a
    # connection opened by a dependency is used from a different thread than
    # it was created in. Safe here because each request gets its own
    # connection and a single request never touches it from two threads at
    # once -- it would not be safe for a connection shared between requests.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # rows accessible by column name
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets the API read while an ingestion job writes, instead of
    # serialising everything behind a single write lock.
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply any migrations the database hasn't seen. Returns the new version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]

    for version, script in enumerate(MIGRATIONS, start=1):
        if version <= current:
            continue
        conn.executescript(script)
        # PRAGMA can't be parameterised, and `version` is a loop index, not input.
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()

    return conn.execute("PRAGMA user_version").fetchone()[0]
