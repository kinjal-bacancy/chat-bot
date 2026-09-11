"""Spreadsheet extraction: XLSX, XLSM, CSV, TSV.

The unit that matters in a spreadsheet is the row. A row is one record --
one gem, one line item -- and its cells only mean anything together. So each
row is emitted as a labelled block:

    Gem: redis
    Version: ~> 4.8
    Status: Version pin

Verbose, but every block is self-contained, which is exactly what retrieval
needs. Exporting the same sheet to PDF is the failure mode this avoids: a
wide table gets paginated by column, and the cells of a row end up on
different pages with nothing connecting them.
"""

import csv
import datetime as dt
import io
from pathlib import Path
from typing import Any, Iterator

import openpyxl

from app.core.parsers.base import NoExtractableText, Page, ParsedDocument, normalise

Row = list[str]

# A table needs at least this many filled cells in a row, on at least two
# consecutive rows, before it is treated as tabular rather than as prose.
_MIN_TABLE_COLUMNS = 2


def _cell_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        # openpyxl reads every number as a float; 63.0 should read as 63.
        return str(int(value))
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value).strip()


def _filled(row: Row) -> int:
    return sum(1 for cell in row if cell)


def _table_regions(rows: list[Row]) -> list[tuple[int, int]]:
    """Find (start, end) row spans that look like tables.

    A sheet is rarely one clean table. Titles, notes and several separate
    tables share a sheet, so regions are detected rather than assumed: a
    table starts where two consecutive rows each have several filled cells,
    and ends at the first row that does not.
    """
    regions: list[tuple[int, int]] = []
    index = 0

    while index < len(rows) - 1:
        starts_table = (
            _filled(rows[index]) >= _MIN_TABLE_COLUMNS
            and _filled(rows[index + 1]) >= _MIN_TABLE_COLUMNS
        )
        if not starts_table:
            index += 1
            continue

        end = index + 1
        while end < len(rows) and _filled(rows[end]) >= _MIN_TABLE_COLUMNS:
            end += 1
        regions.append((index, end))
        index = end

    return regions


def _record_to_text(headers: Row, row: Row) -> str:
    """One row rendered as labelled lines, skipping empty cells."""
    lines = []
    for position, value in enumerate(row):
        if not value:
            continue
        label = headers[position] if position < len(headers) else ""
        lines.append(f"{label}: {value}" if label else value)
    return "\n".join(lines)


def rows_to_text(title: str, rows: list[Row]) -> str:
    """Render a sheet: prose rows as-is, table rows as labelled records."""
    regions = dict(_table_regions(rows))
    blocks: list[str] = []

    index = 0
    while index < len(rows):
        if index in regions:
            end = regions[index]
            headers = rows[index]
            blocks.extend(
                text
                for row in rows[index + 1 : end]
                if (text := _record_to_text(headers, row))
            )
            index = end
            continue

        # Outside a table: a title, a note, a caption. Keep it as prose.
        if line := " ".join(cell for cell in rows[index] if cell).strip():
            blocks.append(line)
        index += 1

    if not blocks:
        return ""  # an empty sheet, not a page with only a heading on it

    if title:
        blocks.insert(0, f"## {title}")

    # Blank lines between blocks so the chunker can see record boundaries.
    return normalise("\n\n".join(blocks))


def _sheet_rows(worksheet) -> list[Row]:
    return [
        [_cell_to_text(cell) for cell in row]
        for row in worksheet.iter_rows(values_only=True)
    ]


def _parse_workbook(path: Path) -> Iterator[tuple[str, list[Row]]]:
    # data_only reads the cached result of a formula rather than "=SUM(B2:B9)",
    # which is what a reader of the sheet actually sees.
    workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        for worksheet in workbook.worksheets:
            yield worksheet.title, _sheet_rows(worksheet)
    finally:
        workbook.close()


def _parse_delimited(path: Path) -> Iterator[tuple[str, list[Row]]]:
    raw = path.read_bytes().decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:8192], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # a single-column file sniffs as nothing
    rows = [
        [cell.strip() for cell in row] for row in csv.reader(io.StringIO(raw), dialect)
    ]
    yield "", rows


def parse(path: Path) -> ParsedDocument:
    reader = _parse_delimited if path.suffix.lower() in {".csv", ".tsv"} else _parse_workbook

    # One page per sheet, so a citation can name where a record came from.
    pages = [
        Page(number=number, text=text)
        for number, (title, rows) in enumerate(reader(path), start=1)
        if (text := rows_to_text(title, rows))
    ]

    if not pages:
        raise NoExtractableText("The spreadsheet contains no data.")

    return ParsedDocument(pages=pages)
