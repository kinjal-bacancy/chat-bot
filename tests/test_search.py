"""Retrieval mechanics.

The fake provider's vectors carry no meaning, so these test plumbing --
ordering, limits, filtering, isolation -- not retrieval quality. Quality is
what the evaluation set in step 11 is for.
"""

from fastapi.testclient import TestClient

from tests.fakes import FakeEmbeddingProvider
from tests.test_parsing import ingest

DOC = b"\n\n".join(
    f"Paragraph number {n}. It carries enough text to take up meaningful room "
    f"in a chunk, so that the document spans more than one of them.".encode()
    for n in range(30)
)


def index(client: TestClient, name: str = "a.txt", content: bytes = DOC) -> str:
    document_id = ingest(client, name, content)
    client.post(f"/documents/{document_id}/parse")
    client.post(f"/documents/{document_id}/chunk")
    client.post(f"/documents/{document_id}/embed")
    return document_id


def search(client: TestClient, query: str = "a question", **body):
    return client.post("/search", json={"query": query, **body})


def test_search_returns_scored_hits(client: TestClient, embeddings: FakeEmbeddingProvider):
    index(client)

    body = search(client).json()

    assert body["hits"]
    assert body["searched_chunks"] > 1
    assert all(-1.0 <= hit["score"] <= 1.0 for hit in body["hits"])


def test_hits_are_ordered_by_descending_score(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    index(client)

    scores = [hit["score"] for hit in search(client).json()["hits"]]

    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_results(client: TestClient, embeddings: FakeEmbeddingProvider):
    index(client)

    assert len(search(client, top_k=2).json()["hits"]) == 2


def test_top_k_larger_than_the_corpus_is_not_an_error(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    index(client)

    body = search(client, top_k=50).json()

    assert len(body["hits"]) == body["searched_chunks"]


def test_hits_carry_their_provenance(client: TestClient, embeddings: FakeEmbeddingProvider):
    """Everything a citation needs: which file, which page, which chunk."""
    document_id = index(client, name="audit.txt")

    hit = search(client).json()["hits"][0]

    assert hit["document_id"] == document_id
    assert hit["filename"] == "audit.txt"
    assert hit["page_number"] >= 1
    assert hit["chunk_id"] and hit["text"]


def test_search_can_be_restricted_to_one_document(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    first = index(client, name="first.txt")
    index(client, name="second.txt", content=DOC.replace(b"Paragraph", b"Section"))

    body = search(client, document_ids=[first]).json()

    assert {hit["document_id"] for hit in body["hits"]} == {first}


def test_searching_an_empty_index_says_nothing_was_searched(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    """Zero results and zero searched chunks are different problems: one is
    'nothing matched', the other is 'you have not indexed anything'."""
    body = search(client).json()

    assert body["hits"] == []
    assert body["searched_chunks"] == 0


def test_unembedded_chunks_are_not_searchable(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    document_id = ingest(client, "a.txt", DOC)
    client.post(f"/documents/{document_id}/parse")
    client.post(f"/documents/{document_id}/chunk")  # deliberately not embedded

    assert search(client).json()["searched_chunks"] == 0


def test_chunks_embedded_by_another_model_are_excluded(
    client: TestClient, app, embeddings: FakeEmbeddingProvider
):
    """Vectors from a different model are points in a different space;
    comparing against them would return confident nonsense."""
    from app.api.deps import get_embedding_provider

    index(client)

    other = FakeEmbeddingProvider()
    other.model = "different-model"
    app.dependency_overrides[get_embedding_provider] = lambda: other

    assert search(client).json()["searched_chunks"] == 0


def test_min_score_filters_weak_matches(client: TestClient, embeddings: FakeEmbeddingProvider):
    index(client)

    assert search(client).json()["hits"], "the default must filter nothing"
    assert search(client, min_score=0.999).json()["hits"] == []


def test_empty_query_is_rejected(client: TestClient, embeddings: FakeEmbeddingProvider):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_deleting_a_document_removes_it_from_search(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    document_id = index(client)
    assert search(client).json()["searched_chunks"] > 0

    client.delete(f"/documents/{document_id}")

    assert search(client).json()["searched_chunks"] == 0
