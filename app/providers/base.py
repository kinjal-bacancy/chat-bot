"""Provider interfaces.

The pipeline depends on these, never on a vendor SDK, so swapping Gemini for
something else is a change confined to this package.
"""

from typing import Protocol, runtime_checkable


class ProviderError(Exception):
    """A provider could not fulfil a request."""


class MissingCredentials(ProviderError):
    """No API key configured for the selected provider."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors.

    Documents and queries are embedded through separate methods on purpose.
    Retrieval models are trained asymmetrically: a question and the passage
    answering it are not the same kind of text, and telling the model which
    one it is looking at measurably improves matching. A single embed()
    would quietly lose that.
    """

    model: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages to be stored and searched over."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a question being asked."""
        ...
