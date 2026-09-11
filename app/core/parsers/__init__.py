"""Document parsing: bytes on disk to text a chunker can work with.

One parser per format, all returning the same `ParsedDocument`, so every
later stage of the pipeline is indifferent to what was uploaded.
"""

from pathlib import Path
from typing import Callable

from app.core.parsers import docx, html, pdf, spreadsheet, text
from app.core.parsers.base import (
    NoExtractableText,
    Page,
    ParsedDocument,
    ParserError,
    normalise,
)

__all__ = [
    "NoExtractableText",
    "Page",
    "ParsedDocument",
    "ParserError",
    "normalise",
    "parse",
    "PARSERS",
]

# Keys must stay in step with storage.ALLOWED_EXTENSIONS; the test suite
# asserts the two agree, so an accepted upload can never hit a missing parser.
PARSERS: dict[str, Callable[[Path], ParsedDocument]] = {
    ".pdf": pdf.parse,
    ".docx": docx.parse,
    ".txt": text.parse,
    ".md": text.parse,
    ".html": html.parse,
    ".htm": html.parse,
    ".xlsx": spreadsheet.parse,
    ".xlsm": spreadsheet.parse,
    ".csv": spreadsheet.parse,
    ".tsv": spreadsheet.parse,
}


def parse(path: Path, extension: str) -> ParsedDocument:
    """Parse a stored file. Raises ParserError for anything the user can fix."""
    parser = PARSERS.get(extension.lower())
    if parser is None:
        raise ParserError(f"No parser for '{extension}'")

    try:
        return parser(path)
    except ParserError:
        raise
    except Exception as exc:
        # Corrupt and malformed files fail in library-specific ways. Wrap them
        # so callers have one exception type to handle.
        raise ParserError(f"Could not parse the file: {exc}") from exc
