"""Getting uploaded bytes safely onto disk.

Nothing here knows about HTTP. Failures raise the exceptions below and the
API layer decides what status code each one deserves.
"""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Extension -> the content type we record. Keyed on extension because the
# browser-supplied Content-Type is attacker-controlled and, for .md in
# particular, inconsistent across browsers anyway.
ALLOWED_EXTENSIONS: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
}

_CHUNK_BYTES = 1024 * 1024  # read the stream 1 MiB at a time


class UnsupportedFileType(Exception):
    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(f"Unsupported file type: {extension or '(none)'}")


class UploadTooLarge(Exception):
    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = limit_bytes
        super().__init__(f"File exceeds the {limit_bytes} byte limit")


class EmptyUpload(Exception):
    pass


class _AsyncReadable(Protocol):
    """The slice of Starlette's UploadFile that we actually depend on."""

    async def read(self, size: int) -> bytes: ...


@dataclass(frozen=True)
class StoredFile:
    id: str
    extension: str
    content_type: str
    size_bytes: int
    sha256: str
    path: Path


def normalise_extension(filename: str | None) -> str:
    """Extract a validated, lowercased extension from an uploaded filename.

    The filename is never used to build a path -- only to pick an extension
    from the allowlist above. That is what makes a filename of
    `../../../etc/passwd` harmless: nothing but the suffix survives, and an
    unlisted suffix is rejected outright.
    """
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileType(suffix)
    return suffix


async def save_upload(
    source: _AsyncReadable,
    filename: str | None,
    *,
    uploads_dir: Path,
    max_bytes: int,
) -> StoredFile:
    """Stream an upload to disk under a generated ID.

    Streamed rather than read whole so that a 5 GB upload is refused after
    the first megabyte over the limit instead of exhausting memory. Written
    to a `.part` file and renamed only on success, so a failed or oversized
    upload never leaves a file that looks complete.
    """
    extension = normalise_extension(filename)

    document_id = uuid.uuid4().hex
    uploads_dir.mkdir(parents=True, exist_ok=True)
    final_path = uploads_dir / f"{document_id}{extension}"
    partial_path = final_path.with_suffix(final_path.suffix + ".part")

    digest = hashlib.sha256()
    size = 0

    try:
        with partial_path.open("wb") as out:
            while chunk := await source.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise UploadTooLarge(max_bytes)
                digest.update(chunk)
                out.write(chunk)

        if size == 0:
            raise EmptyUpload("Uploaded file is empty")

        partial_path.replace(final_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    return StoredFile(
        id=document_id,
        extension=extension,
        content_type=ALLOWED_EXTENSIONS[extension],
        size_bytes=size,
        sha256=digest.hexdigest(),
        path=final_path,
    )
