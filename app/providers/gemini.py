"""Gemini embeddings."""

import logging
import time

import numpy as np
from google import genai
from google.genai import types

from app.providers.base import ProviderError

logger = logging.getLogger(__name__)

# Task types that tell the model whether it is looking at a passage or a
# question. This asymmetry is most of why a retrieval-tuned model beats a
# general-purpose one.
_DOCUMENT = "RETRIEVAL_DOCUMENT"
_QUERY = "RETRIEVAL_QUERY"

_MAX_ATTEMPTS = 5
_BACKOFF_SECONDS = 2.0


def _is_rate_limited(error: Exception) -> bool:
    text = str(error).upper()
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _normalise(values: list[float]) -> list[float]:
    """Scale a vector to unit length.

    Required: gemini-embedding-001 only returns normalised vectors at its
    native 3072 dimensions. Truncated outputs come back unnormalised, and
    cosine similarity over unnormalised vectors silently ranks by magnitude
    as much as by direction.
    """
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.tolist()
    return (vector / norm).tolist()


class GeminiEmbeddingProvider:
    def __init__(
        self, api_key: str, model: str, dimensions: int, batch_size: int = 32
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self._batch_size = batch_size
        self._client = genai.Client(api_key=api_key)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimensions
        )

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.models.embed_content(
                    model=self.model, contents=texts, config=config
                )
                return [_normalise(item.values) for item in response.embeddings]
            except Exception as exc:
                # The free tier is rate limited per minute, and a re-index of a
                # large document will hit it. Waiting is the correct response;
                # failing the whole run is not.
                if _is_rate_limited(exc) and attempt < _MAX_ATTEMPTS:
                    delay = _BACKOFF_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "Embedding rate limited, retrying in %.0fs (attempt %d/%d)",
                        delay, attempt, _MAX_ATTEMPTS,
                    )
                    time.sleep(delay)
                    continue
                raise ProviderError(f"Gemini embedding failed: {exc}") from exc

        raise ProviderError("Gemini embedding failed: rate limit not cleared")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(texts[start : start + self._batch_size], _DOCUMENT))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], _QUERY)[0]
