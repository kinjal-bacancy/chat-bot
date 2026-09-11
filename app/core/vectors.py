"""Vector storage and similarity.

Vectors live in SQLite as float32 blobs and are searched by brute force with
numpy. At this scale that is the right trade: a few thousand chunks compare
in well under a millisecond, the whole corpus stays in the single database
file already being used, and there is no separate service to run. It is an
honest ceiling rather than a hidden one -- somewhere around a hundred
thousand chunks this wants a real index (FAISS, LanceDB, pgvector).

Cosine similarity reduces to a dot product because every stored vector is
normalised to unit length before it gets here.
"""

import numpy as np

# Fixed on purpose: the blob format is a storage contract, and float64 would
# double the database for no measurable retrieval gain.
DTYPE = np.float32


def to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=DTYPE)


def stack(blobs: list[bytes]) -> np.ndarray:
    """Assemble stored vectors into one matrix for a single batched compare."""
    if not blobs:
        return np.empty((0, 0), dtype=DTYPE)
    return np.vstack([from_blob(blob) for blob in blobs])


def similarities(matrix: np.ndarray, query: list[float]) -> np.ndarray:
    """Cosine similarity of a query against every row of `matrix`."""
    if matrix.size == 0:
        return np.empty(0, dtype=DTYPE)

    vector = np.asarray(query, dtype=DTYPE)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm

    return matrix @ vector
