"""The two UI backends must stay interchangeable.

One talks to FastAPI over HTTP, the other calls app.core in-process for
single-process hosts. The UI branches on neither, so a method appearing on
one and not the other breaks a deployment that is never run locally.
"""

import inspect
import sys
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parent.parent / "ui"
sys.path.insert(0, str(UI))

from api import Api  # noqa: E402
from direct import DirectBackend  # noqa: E402

# Everything the Streamlit app calls on a backend.
REQUIRED = [
    "health", "documents", "upload", "ingest", "delete", "chunks",
    "search", "ask_stream",
]


@pytest.mark.parametrize("name", REQUIRED)
def test_both_backends_expose_the_method(name):
    assert hasattr(Api, name), f"Api is missing {name}"
    assert hasattr(DirectBackend, name), f"DirectBackend is missing {name}"


@pytest.mark.parametrize("name", REQUIRED)
def test_the_signatures_match(name):
    """Parameter names, not just presence: the UI passes several by keyword."""
    http = inspect.signature(getattr(Api, name))
    direct = inspect.signature(getattr(DirectBackend, name))

    assert list(http.parameters) == list(direct.parameters), (
        f"{name} differs: {http} vs {direct}"
    )


def test_both_report_a_base_url():
    """The sidebar shows it, so neither may omit it."""
    assert hasattr(Api, "base_url") or "base_url" in Api.__dataclass_fields__
    assert DirectBackend.base_url == "in-process"


def test_direct_backend_health_matches_the_api_response_shape(client):
    """The UI reads specific keys off health(); the in-process backend must
    supply the same ones rather than a near-miss."""
    from app.config import get_settings

    get_settings.cache_clear()
    over_http = set(client.get("/health").json())
    in_process = set(DirectBackend().health())

    assert over_http <= in_process, f"missing from DirectBackend: {over_http - in_process}"
