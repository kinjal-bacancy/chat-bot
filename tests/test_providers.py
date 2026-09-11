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


# --- honouring the server's requested delay ---------------------------


def test_the_requested_retry_delay_is_read_from_the_error():
    """Gemini says how long to wait. Guessing is strictly worse: a
    five-per-minute quota asked for 45s while doubling from 2s had only
    reached 30s, so the retry gave up while the quota was still recovering."""
    assert gemini._requested_delay(Boom("'retryDelay': '45s'")) == 45.0
    assert gemini._requested_delay(Boom("Please retry in 30.4s.")) == pytest.approx(30.4)


def test_a_missing_delay_falls_back_to_backoff():
    assert gemini._requested_delay(Boom("503 UNAVAILABLE")) is None


def test_an_absurd_requested_delay_is_capped():
    """A quota that resets tomorrow must not park the process until then."""
    assert gemini._requested_delay(Boom("'retryDelay': '86400s'")) == gemini._MAX_WAIT_SECONDS


def test_the_server_delay_is_used_instead_of_backoff(monkeypatch):
    slept = []
    monkeypatch.setattr(gemini.time, "sleep", slept.append)
    attempts = []

    def rate_limited():
        attempts.append(1)
        if len(attempts) < 2:
            raise Boom("429 RESOURCE_EXHAUSTED 'retryDelay': '45s'")
        return "ok"

    assert gemini._call_with_retry(rate_limited, "generation") == "ok"
    assert slept == [45.0], "backoff was used instead of the requested delay"
