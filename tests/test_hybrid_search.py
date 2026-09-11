"""Keyword search and fusion.

The fake embedding provider has no semantics, so dense retrieval here is
effectively random. That makes it a good setting for testing the lexical
half: a chunk found by an exact term was found by BM25, not by luck.
"""

from fastapi.testclient import TestClient

from app.core.retrieval import to_fts_query
from tests.fakes import FakeEmbeddingProvider
from tests.test_parsing import ingest

CORPUS = b"""The quarterly report covers revenue and headcount for the period.

Our deployment uses a tool called zzyzxflux for orchestrating releases.

Customer feedback was collected through interviews and written surveys.

Infrastructure costs rose because of increased storage and bandwidth use."""


def index(client: TestClient, content: bytes = CORPUS, name: str = "a.txt") -> str:
    document_id = ingest(client, name, content)
    client.post(f"/documents/{document_id}/ingest")
    return document_id


def search(client: TestClient, query: str, mode: str = "hybrid", **body):
    return client.post(
        "/search", json={"query": query, "mode": mode, "top_k": 3, **body}
    ).json()


# --- query construction -----------------------------------------------


def test_stopwords_are_dropped_from_the_keyword_query():
    """Left in, BM25 ranks on 'what' and 'is' and buries the rare term that
    actually distinguishes the answer -- measured as a real MRR loss."""
    built = to_fts_query("what is the status of the ckeditor gem")

    assert '"ckeditor"' in built
    assert '"what"' not in built and '"the"' not in built


def test_punctuation_cannot_break_the_match_expression():
    """A raw query goes straight into FTS5's own syntax, where a stray quote
    is a syntax error rather than a search."""
    for query in ['gems "quoted" here', "what about -flag", "a AND b OR (c)", "*"]:
        built = to_fts_query(query)
        assert '"' not in built.replace('"', "", built.count('"'))  # balanced
        assert built.count('"') % 2 == 0


def test_a_query_of_only_stopwords_contributes_nothing():
    assert to_fts_query("what is the of and to") == ""


def test_version_strings_survive_tokenising():
    assert '"6.1.7.10"' in to_fts_query("what version is 6.1.7.10")


# --- keyword retrieval -------------------------------------------------


def test_keyword_search_finds_an_exact_rare_term(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    index(client)

    hits = search(client, "zzyzxflux", mode="keyword")["hits"]

    assert hits
    assert "zzyzxflux" in hits[0]["text"]


def test_keyword_search_reports_its_own_scores(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    index(client)

    hit = search(client, "zzyzxflux", mode="keyword")["hits"][0]

    assert hit["keyword_rank"] == 1
    assert hit["keyword_score"] is not None


def test_keyword_search_with_no_match_returns_nothing(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    index(client)

    assert search(client, "nonexistentterm", mode="keyword")["hits"] == []


# --- fusion ------------------------------------------------------------


def test_hybrid_surfaces_what_keyword_alone_found(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    """The whole point of fusing: a chunk the lexical half found must be able
    to reach the results even when the dense half ranked it poorly."""
    index(client)

    hits = search(client, "zzyzxflux")["hits"]

    assert any("zzyzxflux" in hit["text"] for hit in hits)


def test_hybrid_hits_explain_which_retriever_found_them(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    """A result you cannot attribute to a retriever is a result you cannot
    debug."""
    index(client)

    hits = search(client, "zzyzxflux")["hits"]

    assert any(hit["keyword_rank"] is not None for hit in hits)
    assert any(hit["dense_rank"] is not None for hit in hits)


def test_mode_is_echoed_in_the_response(client: TestClient, embeddings: FakeEmbeddingProvider):
    index(client)

    assert search(client, "anything", mode="dense")["mode"] == "dense"
    assert search(client, "anything", mode="keyword")["mode"] == "keyword"


def test_an_unknown_mode_is_rejected(client: TestClient, embeddings: FakeEmbeddingProvider):
    response = client.post("/search", json={"query": "x", "mode": "magic"})

    assert response.status_code == 422


# --- index consistency -------------------------------------------------


def test_rechunking_does_not_leave_stale_keyword_rows(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    """FTS5 has no foreign keys, so its rows are maintained by hand -- an
    easy place for the keyword index to drift out of step with the chunks."""
    document_id = index(client)
    client.post(f"/documents/{document_id}/ingest")

    hits = search(client, "zzyzxflux", mode="keyword")["hits"]

    assert len(hits) == 1


def test_deleting_a_document_clears_its_keyword_rows(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    document_id = index(client)

    client.delete(f"/documents/{document_id}")

    assert search(client, "zzyzxflux", mode="keyword")["hits"] == []


def test_keyword_search_can_be_restricted_to_one_document(
    client: TestClient, embeddings: FakeEmbeddingProvider
):
    first = index(client, name="first.txt")
    index(client, content=CORPUS.replace(b"zzyzxflux", b"zzyzxflux"), name="second.txt")

    hits = search(client, "zzyzxflux", mode="keyword", document_ids=[first])["hits"]

    assert hits and {h["document_id"] for h in hits} == {first}
