"""Thin HTTP client for the RAG API.

The UI holds no retrieval logic of its own. Everything here is a call to the
backend, which keeps the pipeline testable without a browser and means a
different frontend could replace this one without touching it.
"""

import json
from dataclasses import dataclass
from typing import Any, Iterator

import requests

TIMEOUT = 120


class ApiError(Exception):
    """The API refused or could not be reached."""


@dataclass
class Api:
    base_url: str

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = requests.request(
                method, self._url(path), timeout=TIMEOUT, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc

        if not response.ok:
            detail = response.text
            try:
                detail = response.json().get("detail", detail)
            except ValueError:
                pass
            raise ApiError(str(detail))

        return None if response.status_code == 204 else response.json()

    # --- documents ----------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def upload(self, name: str, data: bytes) -> dict:
        return self._request("POST", "/documents", files={"file": (name, data)})

    def ingest(self, document_id: str) -> dict:
        return self._request("POST", f"/documents/{document_id}/ingest")

    def delete(self, document_id: str) -> None:
        self._request("DELETE", f"/documents/{document_id}")

    def chunks(self, document_id: str) -> list[dict]:
        return self._request("GET", f"/documents/{document_id}/chunks")

    # --- asking -------------------------------------------------------

    def search(self, query: str, **body) -> dict:
        return self._request("POST", "/search", json={"query": query, **body})

    def ask_stream(self, question: str, **body) -> Iterator[tuple[str, dict]]:
        """Yield (event name, payload) from the server-sent event stream."""
        try:
            response = requests.post(
                self._url("/ask/stream"),
                json={"question": question, **body},
                stream=True,
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Could not reach the API: {exc}") from exc

        if not response.ok:
            raise ApiError(response.text)

        event = ""
        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if line.startswith("event: "):
                event = line[7:].strip()
            elif line.startswith("data: "):
                yield event, json.loads(line[6:])
