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


class FakeLLMProvider:
    """Returns a scripted answer and records what it was asked.

    Generation quality cannot be unit tested, but everything around it can:
    that the prompt carries the sources, that citations are parsed and
    validated, and that an uncited answer is treated as a refusal.
    """

    model = "fake-llm"

    def __init__(self, reply: str = "The answer is in the sources [1].") -> None:
        self.reply = reply
        self.system_prompts: list[str] = []
        self.prompts: list[str] = []

    def generate(self, system: str, prompt: str) -> str:
        self.system_prompts.append(system)
        self.prompts.append(prompt)
        return self.reply

    def stream(self, system: str, prompt: str):
        yield self.generate(system, prompt)
