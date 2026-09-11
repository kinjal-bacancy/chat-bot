"""Grounded answering.

Whether the model writes a good answer cannot be unit tested. Everything
around it can: that it is given the sources, that its citations are parsed
and validated, that an uncited answer counts as a refusal, and that it is
never asked to answer with no sources at all.
"""

from fastapi.testclient import TestClient

from app.core import answering
from app.core.answering import Source
from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider
from tests.test_parsing import ingest

CORPUS = b"""Puma is the application server. It is pinned at version 5.6.

The ckeditor gem is stale and its upstream is abandoned.

Redis is used for background job queues and caching."""


def index(client: TestClient, content: bytes = CORPUS) -> str:
    document_id = ingest(client, "audit.txt", content)
    client.post(f"/documents/{document_id}/ingest")
    return document_id


def ask(client: TestClient, question: str = "what is the app server", **body):
    return client.post("/ask", json={"question": question, **body}).json()


def source(number: int) -> Source:
    return Source(
        number=number, chunk_id=f"c{number}", document_id="d", filename="f.txt",
        page_number=1, chunk_index=number, heading=None, text="text", score=0.5,
    )


# --- citation parsing --------------------------------------------------


def test_citations_are_collected_in_order_of_first_mention():
    assert answering.cited_numbers("First [2] then [1] again [2].", 3) == [2, 1]


def test_citations_beyond_the_source_list_are_dropped():
    """A model that invents [7] when three sources were given is pointing at
    nothing, and a citation resolving to nothing is worse than none."""
    assert answering.cited_numbers("Claim [7] and claim [2].", 3) == [2]


def test_invalid_citation_markers_are_removed_from_the_text():
    cleaned = answering.strip_invalid_citations("Real [1] fake [9].", 3)

    assert "[1]" in cleaned
    assert "[9]" not in cleaned


def test_an_answer_with_citations_is_not_a_refusal():
    answer = answering.assemble("Puma [1].", [source(1)], "m")

    assert answer.refused is False
    assert [s.number for s in answer.sources] == [1]


def test_an_answer_without_citations_is_a_refusal():
    """The refusal signal is structural: a grounded claim must cite, so
    nothing cited means nothing was grounded."""
    answer = answering.assemble("The documents do not cover this.", [source(1)], "m")

    assert answer.refused is True
    assert answer.sources == []


def test_an_answer_citing_only_invalid_numbers_is_a_refusal():
    answer = answering.assemble("It is definitely this [9].", [source(1)], "m")

    assert answer.refused is True


# --- the prompt --------------------------------------------------------


def test_prompt_contains_every_source_and_its_location():
    built = answering.build_prompt("q?", [source(1), source(2)])

    assert "[1] (f.txt, page 1)" in built
    assert "[2] (f.txt, page 1)" in built
    assert built.rstrip().endswith("Question: q?")


def test_system_prompt_forbids_outside_knowledge_and_demands_citations():
    assert "only" in answering.SYSTEM_PROMPT.lower()
    assert "cite" in answering.SYSTEM_PROMPT.lower()


# --- through the API ---------------------------------------------------


def test_ask_returns_a_cited_answer(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)

    body = ask(client)

    assert body["refused"] is False
    assert body["answer"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["number"] == 1
    assert body["model"] == "fake-llm"


def test_the_model_is_given_the_retrieved_text(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)

    ask(client)

    assert llm.prompts, "the model was not called"
    assert "Puma" in llm.prompts[0] or "ckeditor" in llm.prompts[0]
    assert "Sources:" in llm.prompts[0]


def test_response_carries_everything_retrieved_not_just_what_was_cited(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    """This is what makes a bad answer diagnosable: if the right passage is
    missing from `retrieved`, retrieval failed; if it is there and uncited,
    the model did."""
    index(client)

    body = ask(client, top_k=3)

    assert len(body["retrieved"]) >= len(body["sources"])
    assert all(s["text"] for s in body["retrieved"])


def test_an_uncited_answer_is_reported_as_a_refusal(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)
    llm.reply = "The documents do not cover this."

    body = ask(client)

    assert body["refused"] is True
    assert body["sources"] == []


def test_nothing_indexed_refuses_without_calling_the_model(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    """Handing the model an empty source list invites it to answer from its
    own knowledge, which is the one thing this system must not do."""
    body = ask(client)

    assert body["refused"] is True
    assert body["searched_chunks"] == 0
    assert llm.prompts == [], "the model should not have been called"
    assert "/ingest" in body["answer"]


def test_provider_failure_is_reported_as_a_gateway_error(
    client: TestClient, app, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    from app.api.deps import get_llm_provider
    from app.providers import ProviderError

    index(client)

    class Broken(FakeLLMProvider):
        def generate(self, system, prompt):
            raise ProviderError("quota exhausted")

    app.dependency_overrides[get_llm_provider] = lambda: Broken()

    response = client.post("/ask", json={"question": "anything"})

    assert response.status_code == 502
    assert "quota exhausted" in response.json()["detail"]


def test_empty_question_is_rejected(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_can_be_restricted_to_one_document(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    first = index(client)
    index(client, CORPUS.replace(b"Puma", b"Unicorn"))

    body = ask(client, document_ids=[first])

    assert {s["document_id"] for s in body["retrieved"]} == {first}
