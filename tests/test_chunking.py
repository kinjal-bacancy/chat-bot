"""Chunking.

The requirement worth protecting: a block is never split. The spreadsheet
work upstream exists so a row survives as one unit, and a chunker that cuts
through the middle of a record throws that away again.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.chunking import chunk_page, chunk_pages
from tests.factories import xlsx_bytes
from tests.test_parsing import ingest

SIZES = dict(target_chars=100, overlap_chars=30, max_chars=200)


def chunks_of(text: str, **overrides):
    return chunk_page(text, 1, **{**SIZES, **overrides})[0]


# --- block integrity ---------------------------------------------------


def test_blocks_are_never_split_across_chunks():
    records = "\n\n".join(f"Gem: gem{n}\nVersion: 1.{n}.0\nStatus: Keep" for n in range(8))

    produced = chunks_of(records)

    assert len(produced) > 1
    for chunk in produced:
        for block in chunk.text.split("\n\n"):
            assert block.count("\n") == 2, f"record was cut: {block!r}"


def test_a_record_is_never_orphaned_from_its_label():
    produced = chunks_of("\n\n".join(f"Gem: gem{n}\nVersion: 1.0" for n in range(10)))

    for chunk in produced:
        for block in chunk.text.split("\n\n"):
            assert block.startswith("Gem: ")


# --- overlap -----------------------------------------------------------


def test_consecutive_chunks_overlap_by_whole_blocks():
    produced = chunks_of("\n\n".join(f"Block number {n} with some text" for n in range(10)))

    first_blocks = produced[0].text.split("\n\n")
    second_blocks = produced[1].text.split("\n\n")
    assert first_blocks[-1] == second_blocks[0]


def test_overlap_stays_within_its_budget():
    produced = chunks_of(
        "\n\n".join(f"Block number {n} with some text" for n in range(10)),
        overlap_chars=0,
    )

    assert produced[0].text.split("\n\n")[-1] != produced[1].text.split("\n\n")[0]


def test_chunk_indexes_are_sequential():
    produced = chunks_of("\n\n".join(f"Block {n} of text here" for n in range(12)))

    assert [c.index for c in produced] == list(range(len(produced)))


# --- headings ----------------------------------------------------------


def test_heading_is_prepended_to_every_chunk_beneath_it():
    """A chunk pulled out of the document must still say what it is about."""
    body = "\n\n".join(f"Detail line number {n} goes here" for n in range(8))

    produced = chunks_of(f"## Gem Audit\n\n{body}")

    assert len(produced) > 1
    assert all(c.text.startswith("Gem Audit\n\n") for c in produced)
    assert all(c.heading == "Gem Audit" for c in produced)


def test_a_new_heading_starts_a_new_chunk():
    """Text under a different heading belongs to a different section."""
    produced = chunks_of("## First\n\nShort body.\n\n## Second\n\nOther body.")

    assert [c.heading for c in produced] == ["First", "Second"]
    assert "Other body." not in produced[0].text


def test_heading_carries_across_a_page_break():
    produced = chunk_pages([(1, "## Audit\n\nOne."), (2, "Two.")], **SIZES)

    assert produced[-1].heading == "Audit"
    assert produced[-1].page_number == 2


# --- oversized blocks --------------------------------------------------


def test_oversized_block_is_split_at_sentence_ends():
    prose = " ".join(f"This is sentence number {n}." for n in range(40))

    produced = chunks_of(prose)

    assert len(produced) > 1
    assert all(len(c.text) <= SIZES["max_chars"] for c in produced)
    # Split on sentence ends, so pieces begin with a capital, not mid-word.
    assert all(c.text[0].isupper() for c in produced)


def test_a_single_unbreakable_sentence_is_still_bounded():
    produced = chunks_of("word" * 200)

    assert produced
    assert all(len(c.text) <= SIZES["max_chars"] for c in produced)


# --- offsets and identity ---------------------------------------------


def test_offsets_point_back_into_the_page():
    text = "First block here.\n\nSecond block here.\n\nThird block here."

    for chunk in chunks_of(text, target_chars=20, overlap_chars=0):
        assert text[chunk.char_start : chunk.char_end].strip()


def test_identical_text_has_identical_hashes():
    """The embedding cache in the next step relies on this."""
    a = chunks_of("Some block of text here.")[0]
    b = chunks_of("Some block of text here.")[0]

    assert a.sha256 == b.sha256


def test_empty_page_produces_no_chunks():
    assert chunks_of("   \n\n  \n") == []


# --- through the API ---------------------------------------------------


def test_chunking_a_spreadsheet_keeps_records_whole(client: TestClient):
    rows = [["Gem", "Version", "Status", "Alternative / note"]] + [
        [f"gem{n}", f"1.{n}.0", "Keep", f"Reasonable upstream, no better option for job {n}."]
        for n in range(30)
    ]
    document_id = ingest(client, "audit.xlsx", xlsx_bytes({"Gem Audit": rows}))
    client.post(f"/documents/{document_id}/parse")

    summary = client.post(f"/documents/{document_id}/chunk").json()
    produced = client.get(f"/documents/{document_id}/chunks").json()

    assert summary["chunk_count"] == len(produced) > 1
    for chunk in produced:
        for block in chunk["text"].split("\n\n"):
            if block.startswith("Gem: "):
                assert "Version: " in block and "Status: " in block
                assert "Alternative / note: " in block


def test_chunking_marks_the_document_chunked(client: TestClient):
    document_id = ingest(client, "a.txt", b"Some content to chunk.")
    client.post(f"/documents/{document_id}/parse")

    client.post(f"/documents/{document_id}/chunk")

    assert client.get(f"/documents/{document_id}").json()["status"] == "chunked"


def test_rechunking_replaces_rather_than_appends(client: TestClient):
    document_id = ingest(client, "a.txt", b"Some content to chunk.")
    client.post(f"/documents/{document_id}/parse")

    first = client.post(f"/documents/{document_id}/chunk").json()
    second = client.post(f"/documents/{document_id}/chunk").json()

    assert first["chunk_count"] == second["chunk_count"]


def test_chunking_before_parsing_explains_what_to_do(client: TestClient):
    document_id = ingest(client, "a.txt", b"content")

    response = client.post(f"/documents/{document_id}/chunk")

    assert response.status_code == 409
    assert "/parse" in response.json()["detail"]


def test_reading_chunks_before_chunking_explains_what_to_do(client: TestClient):
    document_id = ingest(client, "a.txt", b"content")
    client.post(f"/documents/{document_id}/parse")

    response = client.get(f"/documents/{document_id}/chunks")

    assert response.status_code == 409
    assert "/chunk" in response.json()["detail"]


def test_deleting_a_document_removes_its_chunks(client: TestClient):
    document_id = ingest(client, "a.txt", b"content")
    client.post(f"/documents/{document_id}/parse")
    client.post(f"/documents/{document_id}/chunk")

    client.delete(f"/documents/{document_id}")

    assert client.get(f"/documents/{document_id}/chunks").status_code == 404
