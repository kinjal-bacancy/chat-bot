"""Choosing how the UI reaches the pipeline.

Two deployments, one UI. Locally and in the container there is a FastAPI
process to call over HTTP; on a single-process host there is not, and the
pipeline is called in-process instead. Both present the same methods, so
nothing in the UI branches on which is in use.
"""

import os

from api import Api, ApiError

__all__ = ["ApiError", "build_backend", "describe"]

# Set when a FastAPI process is reachable. Absent means single-process.
API_URL_VAR = "RAG_API_URL"


def build_backend(api_url: str | None):
    """HTTP client when an API URL is given, in-process backend otherwise."""
    if api_url:
        return Api(api_url)

    # Imported lazily: the HTTP client must stay usable on a machine where
    # the pipeline's own dependencies are not installed.
    from direct import DirectBackend

    return DirectBackend()


def default_api_url() -> str | None:
    """Where the UI should look for an API, if anywhere.

    An empty value is treated as unset, so a host that injects empty
    environment variables does not silently select HTTP against nothing.
    """
    return os.environ.get(API_URL_VAR, "").strip() or None


def describe(backend) -> str:
    return f"API at {backend.base_url}" if backend.base_url != "in-process" else "in-process"
