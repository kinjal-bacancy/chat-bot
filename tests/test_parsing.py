import pytest
from fastapi.testclient import TestClient

from app.core import parsers
from app.core.storage import ALLOWED_EXTENSIONS
from tests.factories import docx_bytes, pdf_bytes


def ingest(client: TestClient, name: str, content: bytes) -> str:
    """Upload a file and return its document id."""
    response = client.post("/documents", files={"file": (name, content)})
    assert response.status_code == 200, response.text
    return response.json()["id"]


def parse(client: TestClient, name: str, content: bytes):
    return client.post(f"/documents/{ingest(client, name, content)}/parse")


def pages_of(client: TestClient, name: str, content: bytes) -> list[str]:
    document_id = ingest(client, name, content)
    assert client.post(f"/documents/{document_id}/parse").status_code == 200
    return [p["text"] for p in client.get(f"/documents/{document_id}/pages").json()]


# --- registry ---------------------------------------------------------


def test_every_accepted_upload_has_a_parser():
    """Otherwise a file could be accepted at upload and then fail to parse."""
    assert set(parsers.PARSERS) == set(ALLOWED_EXTENSIONS)


# --- plain text and markdown -----------------------------------------


def test_text_is_extracted(client: TestClient):
    assert pages_of(client, "a.txt", b"hello world") == ["hello world"]


def test_markdown_structure_is_preserved(client: TestClient):
    """Headings and list markers are the chunker's main split signal, so they
    must survive parsing rather than being rendered away."""
    source = b"# Title\n\nSome text.\n\n- first\n- second\n"

    text = pages_of(client, "a.md", source)[0]

    assert "# Title" in text
    assert "- first" in text


def test_windows_encoded_text_is_decoded(client: TestClient):
    """0x96 is an en dash in cp1252 and invalid UTF-8, so this only decodes
    correctly if the fallback encoding is tried."""
    text = pages_of(client, "a.txt", b"the cost is 10\x9620 dollars")[0]

    assert "10\u201320 dollars" in text


def test_empty_text_file_is_rejected_at_upload(client: TestClient):
    assert client.post("/documents", files={"file": ("a.txt", b"")}).status_code == 400


# --- html -------------------------------------------------------------


def test_html_drops_scripts_and_navigation(client: TestClient):
    source = b"""<html><body>
        <nav>Home About Contact</nav>
        <script>var tracking = 1;</script>
        <main><p>The actual content.</p></main>
        <footer>Copyright 2026</footer>
    </body></html>"""

    text = pages_of(client, "a.html", source)[0]

    assert "The actual content." in text
    assert "tracking" not in text
    assert "Home About Contact" not in text
    assert "Copyright" not in text


def test_html_block_tags_become_line_breaks(client: TestClient):
    """Without this, adjacent block elements run together into one word."""
    text = pages_of(client, "a.html", b"<html><body><p>one</p><p>two</p></body></html>")[0]

    assert "onetwo" not in text
    assert "one" in text and "two" in text


# --- docx -------------------------------------------------------------


def test_docx_paragraphs_are_extracted(client: TestClient):
    text = pages_of(client, "a.docx", docx_bytes(["First para.", "Second para."]))[0]

    assert "First para." in text
    assert "Second para." in text


def test_docx_tables_are_extracted(client: TestClient):
    """Tables hold the facts people ask about -- prices, dates, specs."""
    content = docx_bytes(["Pricing"], tables=[[["Plan", "Price"], ["Pro", "$40"]]])

    text = pages_of(client, "a.docx", content)[0]

    assert "Plan\tPrice" in text
    assert "Pro\t$40" in text


# --- pdf --------------------------------------------------------------


def test_pdf_page_numbers_are_preserved(client: TestClient):
    content = pdf_bytes([["Page one body"], ["Page two body"], ["Page three body"]])
    document_id = ingest(client, "a.pdf", content)
    client.post(f"/documents/{document_id}/parse")

    pages = client.get(f"/documents/{document_id}/pages").json()

    assert [p["number"] for p in pages] == [1, 2, 3]
    assert "Page two body" in pages[1]["text"]


def test_pdf_running_headers_are_stripped(client: TestClient):
    """A header repeated on every page pollutes every chunk and pulls their
    embeddings together."""
    content = pdf_bytes(
        [["ACME Confidential", f"Unique body text {n}"] for n in range(6)]
    )

    texts = pages_of(client, "a.pdf", content)

    assert all("ACME Confidential" not in text for text in texts)
    assert "Unique body text 3" in texts[3]


def test_repeated_line_in_short_pdf_is_kept(client: TestClient):
    """Two pages is not enough evidence that a repeated line is furniture."""
    content = pdf_bytes([["Shared line", "body one"], ["Shared line", "body two"]])

    texts = pages_of(client, "a.pdf", content)

    assert "Shared line" in texts[0]


def test_scanned_pdf_is_rejected_with_an_explanation(client: TestClient):
    """OCR is a non-goal, so an image-only PDF must fail loudly rather than
    be ingested as an empty document that silently never matches a query."""
    response = parse(client, "scan.pdf", pdf_bytes([[], []]))

    assert response.status_code == 422
    assert "scanned" in response.json()["detail"].lower()


def test_sparse_but_real_pdf_is_accepted(client: TestClient):
    """A title page and a divider are not a scanned document."""
    content = pdf_bytes([["Annual Report"], ["Part One"], ["Revenue grew by 12 percent"]])

    texts = pages_of(client, "sparse.pdf", content)

    assert "Annual Report" in texts[0]


def test_failed_parse_is_recorded_on_the_document(client: TestClient):
    document_id = ingest(client, "scan.pdf", pdf_bytes([[]]))

    client.post(f"/documents/{document_id}/parse")

    document = client.get(f"/documents/{document_id}").json()
    assert document["status"] == "failed"
    assert document["error"]


def test_corrupt_file_fails_cleanly(client: TestClient):
    response = parse(client, "broken.pdf", b"%PDF-1.4 this is not a real pdf")

    assert response.status_code == 422
    assert response.json()["detail"]


# --- lifecycle --------------------------------------------------------


def test_parse_marks_the_document_parsed(client: TestClient):
    document_id = ingest(client, "a.txt", b"some content")

    body = client.post(f"/documents/{document_id}/parse").json()

    assert body["status"] == "parsed"
    assert body["page_count"] == 1
    assert body["character_count"] == 12
    assert client.get(f"/documents/{document_id}").json()["status"] == "parsed"


def test_reparsing_replaces_rather_than_appends(client: TestClient):
    """A parser fix should be applied by parsing again, not re-uploading."""
    document_id = ingest(client, "a.pdf", pdf_bytes([["first page of the document"], ["second page of the document"]]))

    client.post(f"/documents/{document_id}/parse")
    client.post(f"/documents/{document_id}/parse")

    assert len(client.get(f"/documents/{document_id}/pages").json()) == 2


def test_deleting_a_document_removes_its_pages(client: TestClient):
    document_id = ingest(client, "a.txt", b"content")
    client.post(f"/documents/{document_id}/parse")

    client.delete(f"/documents/{document_id}")

    assert client.get(f"/documents/{document_id}/pages").status_code == 404


def test_parsing_an_unknown_document_is_404(client: TestClient):
    assert client.post("/documents/nope/parse").status_code == 404
    assert client.get("/documents/nope/pages").status_code == 404


def test_reading_pages_before_parsing_explains_what_to_do(client: TestClient):
    """An empty list here would be indistinguishable from a document that
    parsed fine and contained nothing."""
    document_id = ingest(client, "a.txt", b"some content")

    response = client.get(f"/documents/{document_id}/pages")

    assert response.status_code == 409
    assert "/parse" in response.json()["detail"]


def test_reading_pages_after_a_failed_parse_surfaces_the_error(client: TestClient):
    document_id = ingest(client, "scan.pdf", pdf_bytes([[]]))
    client.post(f"/documents/{document_id}/parse")

    response = client.get(f"/documents/{document_id}/pages")

    assert response.status_code == 409
    assert "scanned" in response.json()["detail"].lower()
