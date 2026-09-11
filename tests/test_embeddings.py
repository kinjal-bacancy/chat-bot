"""Embedding chunks and caching the result."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.core import vectors
from app.providers import MissingCredentials, build_embedding_provider
from tests.fakes import FakeEmbeddingProvider
from tests.test_parsing import ingest

# Long enough to produce several chunks at the default 1200-character target.
DOC = b"\n\n".join(
    f"Paragraph number {n}. It carries enough text to take up meaningful room "
    f"in a chunk, so that the document spans more than one of them.".encode()
    for n in range(30)
)


def prepare(client: TestClient, content: bytes = DOC) -> str:
    document_id = ingest(client, "a.txt", content)
    client.post(f"/documents/{document_id}/parse")
    client.post(f"/documents/{document_id}/chunk")
    return document_id


# --- vector maths ------------------------------------------------------


def test_blob_round_trip_preserves_the_vector():
    original = [0.1, -0.25, 0.5]

    restored = vectors.from_blob(vectors.to_blob(original))

    assert np.allclose(restored, original, atol=1e-6)


def test_identical_vectors_are_maximally_similar():
    matrix = vectors.stack([vectors.to_blob([1.0, 0.0, 0.0])])

    assert vectors.similarities(matrix, [1.0, 0.0, 0.0])[0] == pytest.approx(1.0)


def test_orthogonal_vectors_are_unrelated():
    matrix = vectors.stack([vectors.to_blob([1.0, 0.0])])

    assert vectors.similarities(matrix, [0.0, 1.0])[0] == pytest.approx(0.0)


def test_query_is_normalised_before_comparison():
    """An unnormalised query would rank by magnitude as well as direction."""
    matrix = vectors.stack([vectors.to_blob([1.0, 0.0])])

    assert vectors.similarities(matrix, [50.0, 0.0])[0] == pytest.approx(1.0)


def test_similarity_against_an_empty_store_is_empty():
    assert vectors.similarities(vectors.stack([]), [1.0, 0.0]).size == 0


# --- embedding a document ---------------------------------------------


def test_embedding_covers_every_chunk(client: TestClient, embeddings: FakeEmbeddingProvider):
    document_id = prepare(client)

    body = client.post(f"/documents/{document_id}/embed").json()

    assert body["status"] == "indexed"
    assert body["chunk_count"] > 1
    assert body["embedded"] == body["chunk_count"]
    assert body["reused"] == 0
    assert body["model"] == "fake-embedding"


def test_document_is_marked_indexed(client: TestClient, embeddings: FakeEmbeddingProvider):
    document_id = prepare(client)

    client.post(f"/documents/{document_id}/embed")

    assert client.get(f"/documents/{document_id}").json()["status"] == "indexed"


# --- the cache ---------------------------------------------------------


def test_re_embedding_reuses_the_cache(client: TestClient, embeddings: FakeEmbeddingProvider):
    document_id = prepare(client)
    client.post(f"/documents/{document_id}/embed")
    calls_after_first = embeddings.texts_embedded

    body = client.post(f"/documents/{document_id}/embed").json()

    assert body["embedded"] == 0
    assert body["reused"] == body["chunk_count"]
    assert embeddings.texts_embedded == calls_after_first, "provider was called again"


def test_rechunking_only_embeds_text_that_changed(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    """The point of hashing chunk text rather than chunk ids: re-chunking
    produces new chunk rows, but mostly the same text."""
    document_id = prepare(client)
    client.post(f"/documents/{document_id}/embed")
    before = embeddings.texts_embedded

    client.post(f"/documents/{document_id}/chunk")  # new chunk rows, same text
    body = client.post(f"/documents/{document_id}/embed").json()

    assert body["embedded"] == 0
    assert embeddings.texts_embedded == before


def test_repeated_text_is_embedded_once(client: TestClient, embeddings: FakeEmbeddingProvider):
    """Overlap and boilerplate produce byte-identical chunks."""
    repeated = b"\n\n".join([b"The very same paragraph repeated."] * 6)
    document_id = prepare(client, repeated)

    client.post(f"/documents/{document_id}/embed")

    assert embeddings.texts_embedded == 1


def test_cache_is_keyed_by_model(client: TestClient, app, embeddings: FakeEmbeddingProvider):
    """Switching models must not silently reuse the old model's vectors --
    they are points in a different space."""
    from app.api.deps import get_embedding_provider

    document_id = prepare(client)
    client.post(f"/documents/{document_id}/embed")

    other = FakeEmbeddingProvider()
    other.model = "different-model"
    app.dependency_overrides[get_embedding_provider] = lambda: other

    body = client.post(f"/documents/{document_id}/embed").json()

    assert body["embedded"] == body["chunk_count"]
    assert body["reused"] == 0


# --- failure modes -----------------------------------------------------


def test_embedding_before_chunking_explains_what_to_do(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    document_id = ingest(client, "a.txt", b"content")
    client.post(f"/documents/{document_id}/parse")

    response = client.post(f"/documents/{document_id}/embed")

    assert response.status_code == 409
    assert "/chunk" in response.json()["detail"]


def test_embedding_an_unknown_document_is_404(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    assert client.post("/documents/nope/embed").status_code == 404


def test_provider_failure_is_recorded_and_reported(
    client: TestClient, app, embeddings: FakeEmbeddingProvider
):
    from app.api.deps import get_embedding_provider
    from app.providers import ProviderError

    document_id = prepare(client)

    class Broken(FakeEmbeddingProvider):
        def embed_documents(self, texts):
            raise ProviderError("quota exhausted")

    app.dependency_overrides[get_embedding_provider] = lambda: Broken()

    response = client.post(f"/documents/{document_id}/embed")

    assert response.status_code == 502
    assert "quota exhausted" in response.json()["detail"]
    assert client.get(f"/documents/{document_id}").json()["status"] == "failed"


def test_missing_api_key_names_the_variable_and_where_to_get_one():
    with pytest.raises(MissingCredentials) as raised:
        build_embedding_provider(Settings(gemini_api_key=None))

    assert "GEMINI_API_KEY" in str(raised.value)
    assert "aistudio.google.com" in str(raised.value)
