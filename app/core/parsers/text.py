"""Plain text and Markdown.

Markdown is deliberately left as-is rather than rendered. Its headings and
list markers are exactly the structure the chunker uses to find sensible
split points, so stripping them would discard the most useful signal in the
document.
"""

from pathlib import Path

from app.core.parsers.base import NoExtractableText, Page, ParsedDocument, normalise

# Tried in order. Most files are UTF-8; cp1252 catches Windows-authored text
# whose smart quotes would otherwise decode as mojibake.
_ENCODINGS = ("utf-8", "cp1252")


def _decode(raw: bytes) -> str:
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # Last resort: never fail the whole document over a few bad bytes.
    return raw.decode("utf-8", errors="replace")


def parse(path: Path) -> ParsedDocument:
    text = normalise(_decode(path.read_bytes()))

    if not text:
        raise NoExtractableText("The file contains no text.")

    return ParsedDocument(pages=[Page(number=1, text=text)])
