"""Gemini embeddings."""

import logging
import time
from typing import Iterator

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


# Transient conditions worth waiting out rather than surfacing. 429 is the
# free tier's per-minute limit; 503 is Gemini shedding load under demand and
# is just as temporary. Both were hit within minutes of each other in
# ordinary use, and a user retrying by hand is not an error-handling strategy.
_RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL")


def _is_retryable(error: Exception) -> bool:
    text = str(error).upper()
    return any(marker in text for marker in _RETRYABLE)


def _call_with_retry(operation, describe: str):
    """Run a Gemini call, waiting out rate limits.

    The free tier is limited per minute, and both embedding a document and
    answering a question can hit it. Waiting is the correct response; failing
    the user's request is not.
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            if not _is_retryable(exc):
                raise ProviderError(f"Gemini {describe} failed: {exc}") from exc

            if attempt == _MAX_ATTEMPTS:
                # Say that it was retried, so a persistent outage is not read
                # as a one-off blip that might work if you just try again.
                raise ProviderError(
                    f"Gemini {describe} failed after {_MAX_ATTEMPTS} attempts: {exc}"
                ) from exc

            delay = _BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s hit a transient error, retrying in %.0fs (attempt %d/%d): %s",
                describe, delay, attempt, _MAX_ATTEMPTS, exc,
            )
            time.sleep(delay)


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

        response = _call_with_retry(
            lambda: self._client.models.embed_content(
                model=self.model, contents=texts, config=config
            ),
            "embedding",
        )
        return [_normalise(item.values) for item in response.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed(texts[start : start + self._batch_size], _DOCUMENT))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], _QUERY)[0]


class GeminiLLMProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        thinking_budget: int = 0,
    ) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)
        # Temperature 0 by default: the job is to restate what the sources
        # say, and sampling variety here shows up as invention.
        #
        # Thinking is off by default, and that is a correctness fix as much
        # as a latency one. Thinking tokens are drawn from max_output_tokens,
        # so a long answer can spend the whole budget reasoning and return
        # nothing at all -- measured here, a 200-word request came back empty.
        # Restating retrieved passages is not a task that needs deliberation.
        self._config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        )

    def _with_system(self, system: str) -> types.GenerateContentConfig:
        return self._config.model_copy(update={"system_instruction": system})

    def generate(self, system: str, prompt: str) -> str:
        response = _call_with_retry(
            lambda: self._client.models.generate_content(
                model=self.model, contents=prompt, config=self._with_system(system)
            ),
            "generation",
        )

        if not response.text:
            # Never return empty. An empty answer cites nothing, and an
            # answer citing nothing is read as a refusal -- so the user would
            # be told the documents do not cover their question when in fact
            # the model produced nothing.
            reason = (
                response.candidates[0].finish_reason if response.candidates else None
            )
            raise ProviderError(
                f"Gemini returned no text (finish_reason={reason}). "
                f"If this is MAX_TOKENS, raise LLM_MAX_OUTPUT_TOKENS."
            )
        return response.text

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        # Retried only up to the first token: once output has been sent to
        # the caller, restarting would duplicate what they already have.
        stream = _call_with_retry(
            lambda: self._client.models.generate_content_stream(
                model=self.model, contents=prompt, config=self._with_system(system)
            ),
            "generation",
        )
        produced = False
        try:
            for chunk in stream:
                if chunk.text:
                    produced = True
                    yield chunk.text
        except Exception as exc:
            raise ProviderError(f"Gemini generation failed: {exc}") from exc

        if not produced:
            raise ProviderError(
                "Gemini returned no text. If this recurs, raise "
                "LLM_MAX_OUTPUT_TOKENS or check the model is available."
            )
