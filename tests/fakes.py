"""Test doubles for providers."""

import hashlib

import numpy as np


class FakeEmbeddingProvider:
    """Deterministic embeddings derived from the text's hash.

    Identical text always gives an identical unit vector, and unrelated text
    gives an unrelated one -- enough to exercise caching, storage and cosine
    similarity without a network call. It carries no semantics, so it cannot
    stand in for tests about retrieval *quality*.
    """

    model = "fake-embedding"
    dimensions = 16

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        values = np.random.default_rng(seed).normal(size=self.dimensions)
        return (values / np.linalg.norm(values)).astype(np.float32).tolist()

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.document_calls)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)
