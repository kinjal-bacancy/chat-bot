"""Spreadsheet parsing.

The requirement these tests protect is that a row survives as one unit. A
row's cells only mean anything together, and separating them is precisely
how the PDF export of this data became unusable.
"""

from fastapi.testclient import TestClient

from tests.factories import xlsx_bytes
from tests.test_parsing import ingest, pages_of

AUDIT = [
    ["Category", "Gem", "Version", "Status"],
    ["Framework", "rails", "6.1.7.10", "Keep"],
    ["Framework", "puma", "~> 5.6", "Keep"],
]


def test_a_row_is_kept_together_as_one_block(client: TestClient):
    text = pages_of(client, "audit.xlsx", xlsx_bytes({"Gems": AUDIT}))[0]

    block = next(b for b in text.split("\n\n") if "rails" in b)
    assert "Gem: rails" in block
    assert "Version: 6.1.7.10" in block
    assert "Status: Keep" in block


def test_each_sheet_becomes_its_own_page(client: TestClient):
    content = xlsx_bytes({"Gems": AUDIT, "Notes": [["Heading", "Value"], ["a", "b"]]})
    document_id = ingest(client, "book.xlsx", content)
    client.post(f"/documents/{document_id}/parse")

    pages = client.get(f"/documents/{document_id}/pages").json()

    assert len(pages) == 2
    assert "## Gems" in pages[0]["text"]
    assert "## Notes" in pages[1]["text"]


def test_prose_rows_outside_a_table_are_preserved(client: TestClient):
    """Titles and explanatory notes share a sheet with its tables."""
    content = xlsx_bytes({"S": [["Gemfile Audit"], ["Read every row as a lead."], [], *AUDIT]})

    text = pages_of(client, "a.xlsx", content)[0]

    assert "Read every row as a lead." in text
    assert "Gem: rails" in text


def test_two_tables_on_one_sheet_are_detected_separately(client: TestClient):
    """Each table needs its own header, or the second gets labelled with the
    first one's column names."""
    content = xlsx_bytes({"S": [
        ["Status", "Rows"], ["Keep", 63],
        [],
        ["Category", "Count"], ["Framework", 13],
    ]})

    text = pages_of(client, "a.xlsx", content)[0]

    assert "Status: Keep" in text
    assert "Rows: 63" in text
    assert "Category: Framework" in text
    assert "Count: 13" in text


def test_whole_numbers_are_not_rendered_as_floats(client: TestClient):
    """openpyxl reads every number as a float; 63.0 would read as wrong."""
    text = pages_of(client, "a.xlsx", xlsx_bytes({"S": [["Status", "Rows"], ["Keep", 63]]}))[0]

    assert "Rows: 63" in text
    assert "63.0" not in text


def test_empty_cells_are_skipped(client: TestClient):
    content = xlsx_bytes({"S": [["A", "B", "C"], ["one", None, "three"]]})

    text = pages_of(client, "a.xlsx", content)[0]

    assert "A: one" in text
    assert "C: three" in text
    assert "B:" not in text


def test_empty_spreadsheet_is_rejected(client: TestClient):
    document_id = ingest(client, "a.xlsx", xlsx_bytes({"Blank": []}))

    response = client.post(f"/documents/{document_id}/parse")

    assert response.status_code == 422


def test_csv_is_parsed_as_records(client: TestClient):
    content = b"Gem,Version,Status\nrails,6.1.7.10,Keep\npuma,~> 5.6,Keep\n"

    text = pages_of(client, "a.csv", content)[0]

    block = next(b for b in text.split("\n\n") if "rails" in b)
    assert "Gem: rails" in block
    assert "Version: 6.1.7.10" in block


def test_tsv_is_parsed_as_records(client: TestClient):
    content = b"Gem\tVersion\nrails\t6.1.7.10\n"

    text = pages_of(client, "a.tsv", content)[0]

    assert "Gem: rails" in text
    assert "Version: 6.1.7.10" in text


def test_single_column_csv_still_parses(client: TestClient):
    """A one-column file has no delimiter to sniff."""
    text = pages_of(client, "a.csv", b"alpha\nbeta\ngamma\n")[0]

    assert "alpha" in text and "gamma" in text
