"""Open-access PDF full text for OpenAlex works, next to the abstract.

Only PDF links published by OpenAlex are tried. A failed or missing PDF
leaves the abstract-only document intact and records why in
document.metadata.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
from typing import Any, Callable, Dict, List, Mapping

from ..core.aio import resolve
from ..core.config import load_catalog
from ..core.models import DocumentEnvelope
from .file_adapters import UnsupportedFileFormat, _pdf
from .snapshots import snapshot_bytes

logger = logging.getLogger(__name__)


def require_pdf_support() -> None:
    if importlib.util.find_spec("docling") is None:
        raise UnsupportedFileFormat(
            "PDF full text requires Docling: "
            'pip install -e ".[pdf]" or pass --no-fulltext'
        )


def openalex_pdf_urls(payload: Mapping[str, Any]) -> List[str]:
    """PDF links in OpenAlex order of preference.

    Best OA location first, then the primary location, then the rest.
    """
    locations = [
        payload.get("best_oa_location"),
        payload.get("primary_location"),
        *(payload.get("locations") or []),
    ]
    urls = []
    for location in locations:
        if not isinstance(location, Mapping) or location.get("is_oa") is False:
            continue
        url = location.get("pdf_url")
        if (
            isinstance(url, str)
            and url.startswith(("http://", "https://"))
            and url not in urls
        ):
            urls.append(url)
    return urls[
        : load_catalog("pipeline")["openalex_fulltext"]["max_candidates"]
    ]


def section_role(heading: str | None) -> str | None:
    """Rhetorical role of a paper section (method, results, limitations...).

    The packet stream stays one "fulltext" kind; the role is navigation
    metadata, so a heading never becomes evidence.
    """
    if not heading:
        return None
    for role, pattern in load_catalog("pipeline")["section_roles"].items():
        if re.search(pattern, heading, re.IGNORECASE):
            return role
    return None


def _body_chunks(raw: bytes, document: DocumentEnvelope, url: str) -> tuple:
    settings = load_catalog("pipeline")["openalex_fulltext"]
    snapshot = snapshot_bytes(raw)
    _, chunks, warnings = _pdf(raw, document.document_version_id, snapshot)
    skipped = set(settings["skipped_labels"])
    bibliography = re.compile(settings["bibliography_heading"], re.IGNORECASE)
    kept, heading, in_bibliography = [], None, False
    for chunk in chunks:
        label = chunk.kind
        if label == "section_header":
            heading = chunk.text.strip()
            in_bibliography = bool(bibliography.match(heading))
        if label in skipped or in_bibliography:
            continue
        # One kind keeps a paper in one packet stream; the Docling label
        # stays visible.
        chunk.kind = "fulltext"
        chunk.section_path = ["fulltext"]
        chunk.locator.update(
            docling_label=label,
            section_heading=heading,
            section_role=section_role(heading),
            pdf_url=url,
        )
        kept.append(chunk)
    return snapshot, kept, warnings


async def attach_openalex_fulltext(
    document: DocumentEnvelope,
    payload: Mapping[str, Any],
    fetch: Callable[[str], Any] = None,
) -> DocumentEnvelope:
    """Download and parse the first usable open-access PDF.

    ``fetch`` may be sync or async. Docling runs in a worker thread with a
    timeout, so a pathological PDF fails this document instead of the job.
    """
    if fetch is None:
        from .connectors import fetch_pdf as fetch
    urls = openalex_pdf_urls(payload)
    status: Dict[str, Any] = {
        "status": "no_pdf_url",
        "candidates": urls,
        "attempts": [],
    }
    document.metadata["fulltext"] = status
    if not urls:
        logger.debug(
            "%s: no open-access PDF link", document.document_version_id
        )
        return document
    require_pdf_support()
    for url in urls:
        try:
            raw = await resolve(fetch(url))
            timeout = load_catalog("pipeline")["file_limits"].get(
                "pdf_timeout_seconds"
            )
            snapshot, chunks, warnings = await asyncio.wait_for(
                asyncio.to_thread(_body_chunks, raw, document, url),
                timeout,
            )
        except Exception as exc:
            logger.warning(
                "Full text %s failed: %s: %s", url, type(exc).__name__, exc
            )
            logger.debug("Full text %s traceback", url, exc_info=True)
            status["attempts"].append(
                {"url": url, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            continue
        if not chunks:
            logger.warning("Full text %s has no text chunks", url)
            status["attempts"].append({"url": url, "error": "no_text_chunks"})
            continue
        offset = len(document.chunks)
        for index, chunk in enumerate(chunks):
            chunk.order = offset + index
        document.chunks.extend(chunks)
        document.coverage = "abstract_and_full_text" if offset else "full_text"
        status.update(
            status="parsed",
            pdf_url=url,
            snapshot_uri=snapshot.as_uri(),
            sha256=snapshot.name,
            byte_length=len(raw),
            chunks=len(chunks),
            warnings=warnings,
        )
        logger.debug("Full text %s parsed into %d chunks", url, len(chunks))
        return document
    status["status"] = "failed"
    logger.warning(
        "%s: every PDF candidate failed (%d)",
        document.document_version_id,
        len(urls),
    )
    return document
