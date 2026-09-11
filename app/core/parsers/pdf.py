"""PDF text extraction."""

from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from app.core.parsers.base import NoExtractableText, Page, ParsedDocument, normalise

# Scanned PDFs yield essentially nothing per page, so the test is on the
# document average rather than on any single page -- a real document may well
# have a sparse title page or a near-empty divider, and rejecting it for that
# would be wrong.
_MIN_AVERAGE_CHARS_PER_PAGE = 10
# Repeated-line stripping needs enough pages for "repeated" to mean something.
_MIN_PAGES_FOR_HEADER_DETECTION = 4
# A line must appear on at least this share of pages to count as furniture.
_REPEAT_RATIO = 0.6


def _strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers and footers.

    A line appearing at the top or bottom of most pages is furniture, not
    content. Left in, it is repeated into many chunks, where it both wastes
    context and gives every chunk a block of identical text that drags their
    embeddings toward each other.

    Only the first and last two lines of each page are considered, so a
    sentence that happens to recur in the body is safe.
    """
    if len(pages) < _MIN_PAGES_FOR_HEADER_DETECTION:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        counts.update(set(lines[:2] + lines[-2:]))

    threshold = len(pages) * _REPEAT_RATIO
    furniture = {line for line, count in counts.items() if count >= threshold}
    if not furniture:
        return pages

    cleaned = []
    for page in pages:
        lines = page.splitlines()
        head = 0
        while head < len(lines) and lines[head].strip() in furniture | {""}:
            head += 1
        tail = len(lines)
        while tail > head and lines[tail - 1].strip() in furniture | {""}:
            tail -= 1
        cleaned.append("\n".join(lines[head:tail]))
    return cleaned


def parse(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))

    raw = [normalise(page.extract_text() or "") for page in reader.pages]
    cleaned = _strip_repeated_lines(raw)

    pages = [
        Page(number=index, text=normalise(text))
        for index, text in enumerate(cleaned, start=1)
    ]

    extracted = sum(len(page.text) for page in pages)
    if not pages or extracted / len(pages) < _MIN_AVERAGE_CHARS_PER_PAGE:
        raise NoExtractableText(
            f"Almost no text could be extracted from this PDF "
            f"({extracted} characters across {len(pages)} page(s)). It is "
            f"most likely scanned or image-only, and OCR is not supported."
        )

    # Keep empty pages: dropping them would shift every later page number and
    # break the citation a reader uses to find the passage in the original.
    return ParsedDocument(pages=pages)
