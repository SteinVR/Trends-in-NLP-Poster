"""Shared filesystem hashing helpers."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def sha256_for_file(path: Path, *, chunk_size: int = 1_048_576) -> str:
    """Return the lowercase SHA-256 digest for one file path.

    Args:
        path: Absolute or relative path to the file to hash.
        chunk_size: Read block size in bytes used while streaming the file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
