"""The streaming answer endpoint.

Streaming spends the status code before anything can go wrong, so the
guarantees worth testing are that the event sequence is well formed and that
a mid-stream failure is still reported.
"""

import json

from fastapi.testclient import TestClient

from tests.fakes import FakeEmbeddingProvider, FakeLLMProvider
from tests.test_ask import CORPUS, index


def events(client: TestClient, question: str = "what is the app server", **body):
    """Parse the server-sent event stream into (name, payload) pairs."""
    parsed, name = [], ""
    with client.stream(
        "POST", "/ask/stream", json={"question": question, **body}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("event: "):
                name = line[7:].strip()
            elif line.startswith("data: "):
                parsed.append((name, json.loads(line[6:])))
    return parsed


def test_stream_sends_retrieved_then_deltas_then_done(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)

    names = [name for name, _ in events(client)]

    assert names[0] == "retrieved"
    assert names[-1] == "done"
    assert "delta" in names


def test_sources_arrive_before_the_answer(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    """So a UI can fill its inspector while the answer is still arriving."""
    index(client)

    first_name, first_payload = events(client)[0]

    assert first_name == "retrieved"
    assert first_payload["retrieved"]
    assert first_payload["searched_chunks"] > 0


def test_deltas_reassemble_into_the_final_answer(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)
    llm.reply = "Puma is the app server [1]."

    stream = events(client)
    streamed = "".join(p["text"] for n, p in stream if n == "delta")
    done = next(p for n, p in stream if n == "done")

    assert streamed.strip() == done["answer"]


def test_done_carries_citations_and_refusal_state(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)

    done = next(p for n, p in events(client) if n == "done")

    assert done["refused"] is False
    assert [s["number"] for s in done["sources"]] == [1]


def test_an_uncited_stream_is_reported_as_refused(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    index(client)
    llm.reply = "The documents do not cover this."

    done = next(p for n, p in events(client) if n == "done")

    assert done["refused"] is True
    assert done["sources"] == []


def test_nothing_indexed_completes_without_calling_the_model(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    stream = events(client)
    done = next(p for n, p in stream if n == "done")

    assert done["refused"] is True
    assert llm.prompts == []
    assert not [n for n, _ in stream if n == "delta"]


def test_a_mid_stream_failure_is_reported_as_an_event(
    client: TestClient, app, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    """The status code is already spent by then, so an error event is the
    only way left to tell the client something went wrong."""
    from app.api.deps import get_llm_provider
    from app.providers import ProviderError

    index(client)

    class Broken(FakeLLMProvider):
        def stream(self, system, prompt):
            raise ProviderError("quota exhausted")
            yield  # pragma: no cover

    app.dependency_overrides[get_llm_provider] = lambda: Broken()

    names = [n for n, _ in events(client)]
    payloads = dict(events(client))

    assert "error" in names
    assert "quota exhausted" in payloads["error"]["detail"]


def test_stream_respects_a_document_filter(
    client: TestClient, embeddings: FakeEmbeddingProvider, llm: FakeLLMProvider
):
    first = index(client)
    index(client, CORPUS.replace(b"Puma", b"Unicorn"))

    _, payload = events(client, document_ids=[first])[0]

    assert {s["document_id"] for s in payload["retrieved"]} == {first}
