"""Content-addressed original bytes; a source location remains separate from
its snapshot.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from ..core.config import load_catalog
from ..core.models import Artifact, DocumentEnvelope


def snapshot_bytes(
    raw: bytes, directory: Optional[Union[Path, str]] = None
) -> Path:
    if directory is None:
        directory = os.getenv(
            "LCTREND_RAW_DIR",
            load_catalog("runtime").get("raw_directory", "artifacts/raw"),
        )
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / hashlib.sha256(raw).hexdigest()

    def verify_existing() -> None:
        if path.read_bytes() != raw:
            raise ValueError("Raw snapshot does not match its content hash")

    if path.exists():
        verify_existing()
        return path
    temporary = None
    try:
        descriptor, filename = tempfile.mkstemp(prefix=".snapshot-", dir=root)
        temporary = Path(filename)
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        # A same-filesystem hard link publishes complete bytes in one operation
        # and never replaces an existing hash path. Concurrent writers can only
        # observe a complete published file, never the partially written temp.
        try:
            os.link(temporary, path)
        except FileExistsError:
            verify_existing()
        except OSError:
            # Some mounts (Docker Desktop bind mounts on Windows) refuse hard
            # links. The name is a content hash, so an atomic replace of an
            # identical file is equally safe.
            if path.exists():
                verify_existing()
            else:
                os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def persist_snapshot(
    document: DocumentEnvelope,
    raw: bytes,
    directory: Optional[Union[Path, str]] = None,
) -> DocumentEnvelope:
    path = snapshot_bytes(raw, directory)
    document.artifact = Artifact(
        uri=path.as_uri(),
        sha256=path.name,
        media_type=document.artifact.media_type,
        byte_length=len(raw),
    )
    return document
