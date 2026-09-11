"""Shared types for document parsers."""

import re
from dataclasses import dataclass


class ParserError(Exception):
    """Parsing failed in a way the user can act on."""


class NoExtractableText(ParserError):
    """The file parsed, but yielded no usable text.

    Almost always a scanned or image-only PDF. OCR is an explicit non-goal,
    so these are rejected loudly instead of being ingested as an empty
    document that silently never matches a query.
    """


@dataclass(frozen=True)
class Page:
    """A unit of a document that a citation can point at.

    Formats without real pagination (txt, md, html, docx) produce a single
    page numbered 1; locality within those comes from chunk offsets instead.
    """

    number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    pages: list[Page]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def character_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
# Zero-width and non-breaking characters that PDFs emit freely and that
# otherwise survive into chunks and embeddings as invisible noise.
_INVISIBLES = str.maketrans({" ": " ", "​": "", "﻿": "", "\r": "\n"})


def normalise(text: str) -> str:
    """Tidy extracted text without destroying structure.

    Blank lines are preserved (collapsed to at most one) because paragraph
    boundaries are the main signal the chunker has to work with.
    """
    text = text.translate(_INVISIBLES)
    text = _TRAILING_SPACE.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
