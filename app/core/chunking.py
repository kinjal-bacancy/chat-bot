"""Splitting page text into retrievable chunks.

Two ideas drive the design.

Blocks are atomic. Text is split on blank lines first, and chunks are built
by packing whole blocks. A spreadsheet row rendered as a labelled record is
one block, and half a record -- a version with no gem name -- is worse than
no record at all, because it still retrieves and then misleads.

Chunks carry their heading. A chunk lifted out of a document loses the
context that told you what it was about, so the enclosing heading is
prepended to its text. That context is then part of what gets embedded,
which is what lets a chunk about "~> 5.6" match a question about Puma.
"""

import hashlib
import re
import uuid
from dataclasses import dataclass

# One or more blank lines: the boundary every parser here emits between
# paragraphs, records and list groups.
_BLOCK_SPLIT = re.compile(r"\n\s*\n")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
# Sentence end followed by whitespace. Deliberately simple: it only has to be
# a better split point than an arbitrary character offset.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    index: int
    page_number: int
    heading: str | None
    text: str
    char_start: int
    char_end: int

    @property
    def sha256(self) -> str:
        """Identity of the chunk's text, used to reuse a cached embedding
        when re-chunking leaves a chunk's content unchanged."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def new_id(self) -> str:
        return uuid.uuid4().hex


@dataclass(frozen=True)
class _Block:
    text: str
    start: int
    end: int


def _split_blocks(text: str) -> list[_Block]:
    """Split on blank lines, keeping each block's offsets in the page."""
    blocks: list[_Block] = []
    position = 0
    for piece in _BLOCK_SPLIT.split(text):
        start = text.find(piece, position) if piece else position
        if piece.strip():
            blocks.append(_Block(piece.strip(), start, start + len(piece)))
        position = start + len(piece)
    return blocks


def _split_oversized(block: _Block, limit: int) -> list[_Block]:
    """Break a block too large to be a chunk, preferring sentence ends.

    Only reached by genuinely long prose. Offsets stay relative to the page
    so a citation still points at the right region.
    """
    pieces: list[_Block] = []
    buffer, buffer_start = "", block.start

    for sentence in _SENTENCE_END.split(block.text):
        if buffer and len(buffer) + len(sentence) + 1 > limit:
            pieces.append(_Block(buffer.strip(), buffer_start, buffer_start + len(buffer)))
            buffer_start += len(buffer) + 1
            buffer = ""
        buffer = f"{buffer} {sentence}".strip() if buffer else sentence

        # A single sentence longer than the limit: nothing left but to cut it.
        while len(buffer) > limit:
            pieces.append(_Block(buffer[:limit], buffer_start, buffer_start + limit))
            buffer = buffer[limit:]
            buffer_start += limit

    if buffer:
        pieces.append(_Block(buffer.strip(), buffer_start, buffer_start + len(buffer)))
    return pieces


def _overlap_tail(blocks: list[_Block], budget: int) -> list[_Block]:
    """The trailing blocks that fit in the overlap budget.

    Overlap is whole blocks rather than a fixed number of trailing
    characters, so it never starts a chunk with the tail end of a record.
    """
    tail: list[_Block] = []
    used = 0
    for block in reversed(blocks):
        if used + len(block.text) > budget:
            break
        tail.insert(0, block)
        used += len(block.text)
    return tail


def chunk_page(
    text: str,
    page_number: int,
    *,
    target_chars: int,
    overlap_chars: int,
    max_chars: int,
    start_index: int = 0,
    inherited_heading: str | None = None,
) -> tuple[list[Chunk], str | None]:
    """Chunk one page. Returns its chunks and the heading still in scope.

    The trailing heading is handed back so it carries onto the next page --
    a section that spans a page break should not lose its title halfway
    through.
    """
    chunks: list[Chunk] = []
    heading = inherited_heading
    pending: list[_Block] = []
    pending_heading = heading
    index = start_index

    def flush() -> None:
        nonlocal pending, index
        if not pending:
            return
        body = "\n\n".join(block.text for block in pending)
        chunks.append(
            Chunk(
                index=index,
                page_number=page_number,
                heading=pending_heading,
                text=f"{pending_heading}\n\n{body}" if pending_heading else body,
                char_start=pending[0].start,
                char_end=pending[-1].end,
            )
        )
        index += 1
        pending = _overlap_tail(pending, overlap_chars)

    for block in _split_blocks(text):
        match = _MARKDOWN_HEADING.match(block.text)
        if match:
            # A heading ends the current chunk: text after it belongs to a
            # different section and should not be packed in with what came
            # before.
            flush()
            pending = []
            heading = match.group(2).strip()
            pending_heading = heading
            continue

        for piece in (
            _split_oversized(block, target_chars) if len(block.text) > max_chars else [block]
        ):
            packed = sum(len(b.text) for b in pending)
            if pending and packed + len(piece.text) > target_chars:
                flush()
            pending.append(piece)

    flush()
    # Overlap left `pending` populated; nothing further to emit from it.
    return chunks, heading


def chunk_pages(
    pages: list[tuple[int, str]],
    *,
    target_chars: int,
    overlap_chars: int,
    max_chars: int,
) -> list[Chunk]:
    """Chunk a whole document, numbering chunks across all its pages."""
    chunks: list[Chunk] = []
    heading: str | None = None

    for page_number, text in pages:
        page_chunks, heading = chunk_page(
            text,
            page_number,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
            max_chars=max_chars,
            start_index=len(chunks),
            inherited_heading=heading,
        )
        chunks.extend(page_chunks)

    return chunks
