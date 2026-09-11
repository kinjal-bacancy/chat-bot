from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import MAX_UPLOAD_MB


def upload(client: TestClient, name: str, content: bytes, content_type: str = "text/plain"):
    return client.post("/documents", files={"file": (name, content, content_type)})


def test_upload_stores_document(client: TestClient, uploads_dir: Path):
    response = upload(client, "notes.txt", b"retrieval augmented generation")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["status"] == "uploaded"
    assert body["size_bytes"] == 30
    assert body["duplicate"] is False
    assert len(body["sha256"]) == 64

    # Stored under the generated ID, not the uploaded name.
    assert (uploads_dir / f"{body['id']}.txt").exists()


def test_response_does_not_expose_server_paths(client: TestClient):
    body = upload(client, "notes.txt", b"hello").json()

    assert "stored_path" not in body
    assert not any("path" in field for field in body)


def test_identical_bytes_are_deduplicated(client: TestClient, uploads_dir: Path):
    first = upload(client, "notes.txt", b"same content").json()
    second = upload(client, "copy-of-notes.txt", b"same content").json()

    assert second["duplicate"] is True
    assert second["id"] == first["id"]
    assert len(client.get("/documents").json()) == 1
    # The redundant second copy was cleaned up, not left on disk.
    assert len(list(uploads_dir.glob("*.txt"))) == 1


def test_differing_bytes_are_separate_documents(client: TestClient):
    first = upload(client, "a.txt", b"first").json()
    second = upload(client, "b.txt", b"second").json()

    assert first["id"] != second["id"]
    assert len(client.get("/documents").json()) == 2


def test_unsupported_extension_is_rejected(client: TestClient, uploads_dir: Path):
    response = upload(client, "payload.exe", b"MZ\x90\x00")

    assert response.status_code == 415
    assert ".pdf" in response.json()["detail"]
    assert list(uploads_dir.iterdir()) == []


def test_missing_extension_is_rejected(client: TestClient):
    assert upload(client, "README", b"no extension").status_code == 415


def test_empty_file_is_rejected(client: TestClient, uploads_dir: Path):
    response = upload(client, "empty.txt", b"")

    assert response.status_code == 400
    assert list(uploads_dir.iterdir()) == []


def test_oversized_upload_is_rejected_and_leaves_no_partial_file(
    client: TestClient, uploads_dir: Path
):
    oversized = b"x" * (MAX_UPLOAD_MB * 1024 * 1024 + 1)

    response = upload(client, "big.txt", oversized)

    assert response.status_code == 413
    # Neither a finished file nor the `.part` file it was streamed into.
    assert list(uploads_dir.iterdir()) == []


def test_traversal_filename_cannot_escape_the_uploads_directory(
    client: TestClient, uploads_dir: Path
):
    body = upload(client, "../../../../tmp/evil.txt", b"traversal attempt").json()

    stored = list(uploads_dir.glob("*.txt"))
    assert len(stored) == 1
    assert stored[0].name == f"{body['id']}.txt"
    # The original name is kept as metadata for display, never used as a path.
    assert body["filename"] == "../../../../tmp/evil.txt"
    assert not Path("/tmp/evil.txt").exists()


def test_get_document_by_id(client: TestClient):
    created = upload(client, "notes.txt", b"content").json()

    fetched = client.get(f"/documents/{created['id']}")

    assert fetched.status_code == 200
    assert fetched.json()["id"] == created["id"]


def test_get_unknown_document_is_404(client: TestClient):
    assert client.get("/documents/does-not-exist").status_code == 404


def test_delete_removes_row_and_file(client: TestClient, uploads_dir: Path):
    created = upload(client, "notes.txt", b"content").json()
    stored_file = uploads_dir / f"{created['id']}.txt"
    assert stored_file.exists()

    assert client.delete(f"/documents/{created['id']}").status_code == 204

    assert not stored_file.exists()
    assert client.get(f"/documents/{created['id']}").status_code == 404
    assert client.get("/documents").json() == []


def test_delete_unknown_document_is_404(client: TestClient):
    assert client.delete("/documents/does-not-exist").status_code == 404


def test_deduplication_survives_delete(client: TestClient):
    """Re-uploading after a delete stores the document again, rather than
    matching the hash of a document that no longer exists."""
    first = upload(client, "notes.txt", b"content").json()
    client.delete(f"/documents/{first['id']}")

    second = upload(client, "notes.txt", b"content").json()

    assert second["duplicate"] is False
    assert second["id"] != first["id"]
