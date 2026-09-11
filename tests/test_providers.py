"""Provider-level behaviour that does not need the network."""

import pytest

from app.providers import gemini


class Boom(Exception):
    pass


def test_rate_limits_and_outages_are_both_retryable():
    """429 is the free tier's per-minute cap, 503 is Gemini shedding load.
    Both were hit in ordinary use within minutes of each other."""
    for message in [
        "429 RESOURCE_EXHAUSTED quota exceeded",
        "503 UNAVAILABLE model is experiencing high demand",
        "500 INTERNAL",
    ]:
        assert gemini._is_retryable(Boom(message)), message


def test_real_errors_are_not_retried():
    for message in ["401 UNAUTHENTICATED bad key", "400 INVALID_ARGUMENT"]:
        assert not gemini._is_retryable(Boom(message)), message


def test_a_transient_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Boom("503 UNAVAILABLE")
        return "ok"

    assert gemini._call_with_retry(flaky, "generation") == "ok"
    assert len(attempts) == 3


def test_a_permanent_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)
    attempts = []

    def broken():
        attempts.append(1)
        raise Boom("401 UNAUTHENTICATED")

    with pytest.raises(gemini.ProviderError, match="401"):
        gemini._call_with_retry(broken, "generation")
    assert len(attempts) == 1


def test_giving_up_says_what_was_tried(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)

    def always_busy():
        raise Boom("503 UNAVAILABLE")

    with pytest.raises(gemini.ProviderError, match="attempts"):
        gemini._call_with_retry(always_busy, "embedding")


def test_truncated_embeddings_are_normalised():
    """gemini-embedding-001 returns unit vectors only at its native 3072
    dimensions; cosine over unnormalised ones ranks partly by magnitude."""
    normalised = gemini._normalise([3.0, 4.0])

    assert normalised == pytest.approx([0.6, 0.8])


def test_a_zero_vector_does_not_divide_by_zero():
    assert gemini._normalise([0.0, 0.0]) == [0.0, 0.0]
