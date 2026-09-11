"""HTML extraction."""

from pathlib import Path

from bs4 import BeautifulSoup

from app.core.parsers.base import NoExtractableText, Page, ParsedDocument, normalise

# Navigation, boilerplate and code that runs rather than reads. Left in, these
# contribute chunks that match queries on words no human put there as content.
_NON_CONTENT_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript")

# Tags after which a line break preserves the document's structure.
_BLOCK_TAGS = ("p", "div", "section", "article", "br", "li", "tr",
               "h1", "h2", "h3", "h4", "h5", "h6")


def parse(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(path.read_bytes(), "lxml")

    for tag in soup(_NON_CONTENT_TAGS):
        tag.decompose()

    # Prefer <main>/<article> when present: on a real page that is the
    # difference between the content and the content plus every sidebar.
    root = soup.find("main") or soup.find("article") or soup.body or soup

    for tag in root.find_all(_BLOCK_TAGS):
        tag.append("\n")

    text = normalise(root.get_text())

    if not text:
        raise NoExtractableText("The page contains no readable text.")

    return ParsedDocument(pages=[Page(number=1, text=text)])
