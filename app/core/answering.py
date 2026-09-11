"""Turning retrieved chunks into a grounded answer.

The rule the whole design turns on: an answer must cite. Every claim carries
a [n] pointing at a numbered source, so "no citations" is what a refusal
looks like -- a structural signal rather than a phrase to pattern-match.

That matters because the alternative does not work here. Dense scores on real
data sit in a narrow band, so a question about something entirely absent from
the corpus still retrieves chunks at a respectable similarity. There is no
threshold at which to refuse. The model reading the sources is the only thing
that can tell.
"""

import re
from dataclasses import dataclass

from app.core.retrieval import SearchHit

SYSTEM_PROMPT = """\
You answer questions using only the numbered sources given to you.

Rules:
- Use only what the sources say. Never add outside knowledge, and never fill \
a gap with what is usually true.
- Cite every claim with the source number in square brackets, like [2]. A \
sentence drawing on two sources cites both, like [1][3].
- Quote exact values -- versions, names, statuses, numbers -- character for \
character as the source writes them. Do not tidy or normalise them.
- If the sources do not answer the question, say so plainly in one sentence \
and cite nothing. Do not guess, and do not offer a partial answer as if it \
were complete. Saying the documents do not cover it is a correct answer.
- If the sources only partly answer it, give what they support, then say \
which part is missing.
- Be concise. No preamble, no restating the question.\
"""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Source:
    number: int
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    heading: str | None
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[Source]     # those the answer actually cited
    retrieved: list[Source]   # everything offered to the model
    refused: bool
    model: str


def build_sources(hits: list[SearchHit]) -> list[Source]:
    return [
        Source(
            number=position,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            filename=hit.filename,
            page_number=hit.page_number,
            chunk_index=hit.chunk_index,
            heading=hit.heading,
            text=hit.text,
            score=hit.score,
        )
        for position, hit in enumerate(hits, start=1)
    ]


def build_prompt(question: str, sources: list[Source]) -> str:
    """Lay out the sources, then the question.

    Each source is labelled with its file and page so the model can name
    where something came from, and so a reader can go and check.
    """
    blocks = []
    for source in sources:
        location = f"{source.filename}, page {source.page_number}"
        if source.heading:
            location += f", under '{source.heading}'"
        blocks.append(f"[{source.number}] ({location})\n{source.text}")

    return (
        "Sources:\n\n"
        + "\n\n---\n\n".join(blocks)
        + f"\n\n---\n\nQuestion: {question}"
    )


def cited_numbers(text: str, available: int) -> list[int]:
    """Citation numbers the answer actually used, in order of first mention.

    Numbers outside the range offered are dropped: a model that invents [7]
    when six sources were given is pointing at nothing, and a citation that
    resolves to nothing is worse than none at all.
    """
    seen: list[int] = []
    for match in _CITATION.findall(text):
        number = int(match)
        if 1 <= number <= available and number not in seen:
            seen.append(number)
    return seen


def strip_invalid_citations(text: str, available: int) -> str:
    """Remove citation markers that point outside the source list."""
    return _CITATION.sub(
        lambda m: m.group(0) if 1 <= int(m.group(1)) <= available else "", text
    )


def assemble(raw: str, sources: list[Source], model: str) -> Answer:
    text = strip_invalid_citations(raw.strip(), len(sources))
    used = cited_numbers(text, len(sources))

    return Answer(
        text=text,
        sources=[sources[number - 1] for number in used],
        retrieved=sources,
        # No citations means the model found nothing in the sources to stand
        # on. That is the refusal path, and it is a correct outcome.
        refused=not used,
        model=model,
    )
