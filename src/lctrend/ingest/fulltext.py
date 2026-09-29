"""Open-access PDF text and PubMed abstract fallback for OpenAlex works.

Only PDF links published by OpenAlex are tried. If no text remains, a numeric
PMID supplied by OpenAlex may provide an abstract after its DOI is verified.
Missing source text and failed downloads are recorded in document.metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Collection, Dict, List, Mapping

from ..core.aio import resolve
from ..core.config import load_catalog
from ..core.models import Chunk, DocumentEnvelope, stable_id
from .file_adapters import (
    UnsupportedFileFormat,
    _pdf,
    pdf_body,
    section_role,  # noqa: F401 (public name kept here)
)
from .snapshots import snapshot_bytes

logger = logging.getLogger(__name__)


def require_pdf_support() -> None:
    if importlib.util.find_spec("docling") is None:
        raise UnsupportedFileFormat(
            "PDF full text requires Docling: "
            'pip install -e ".[pdf]" or pass --no-fulltext'
        )


def openalex_pdf_urls(payload: Mapping[str, Any]) -> List[str]:
    """PDF links to try, open repositories first.

    OpenAlex order (best OA location, primary, the rest) within three
    tiers: preferred hosts (arXiv, PMC, repositories), others, and
    publishers that refuse automated downloads (pipeline.json
    openalex_fulltext hosts).
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
    settings = load_catalog("pipeline")["openalex_fulltext"]
    preferred = settings.get("preferred_hosts", [])
    blocked = settings.get("blocked_hosts", [])

    def tier(url: str) -> int:
        host = url.split("/")[2].casefold()
        if any(name in host or name in url for name in preferred):
            return 0
        return 2 if any(name in host for name in blocked) else 1

    # sorted is stable: OpenAlex order holds within a tier.
    return sorted(urls, key=tier)[: settings["max_candidates"]]


def _body_chunks(raw: bytes, document: DocumentEnvelope, url: str) -> tuple:
    snapshot = snapshot_bytes(raw)
    _, chunks, warnings = _pdf(raw, document.document_version_id, snapshot)
    return snapshot, pdf_body(chunks, pdf_url=url), warnings


def _numeric_pmid(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"(?:https://pubmed\.ncbi\.nlm\.nih\.gov/)?([1-9][0-9]{0,11})/?",
        value.strip(),
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _pubmed_abstract(
    raw: bytes, pmid: str, doi: str
) -> tuple[str | None, str]:
    # The response is size-bounded by the connector. Reject internal entity
    # declarations before parsing XML supplied by a remote source.
    if b"<!ENTITY" in raw.upper():
        return None, "invalid_xml"
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None, "invalid_xml"
    if root.tag != "PubmedArticleSet" or not root.findall("./PubmedArticle"):
        return None, "invalid_response"
    for article in root.findall(".//PubmedArticle"):
        if (article.findtext("./MedlineCitation/PMID") or "").strip() != pmid:
            continue
        article_dois = {
            (item.text or "").strip().casefold()
            for item in article.findall(".//ArticleId[@IdType='doi']")
            + article.findall(".//ELocationID[@EIdType='doi']")
        }
        if doi.casefold() not in article_dois:
            return None, "doi_mismatch"
        parts = []
        for item in article.findall(
            "./MedlineCitation/Article/Abstract/AbstractText"
        ):
            content = " ".join("".join(item.itertext()).split())
            if content:
                label = (item.get("Label") or "").strip()
                parts.append(f"{label}: {content}" if label else content)
        if not parts:
            return None, "no_abstract"
        return "\n\n".join(parts), "parsed"
    return None, "pmid_mismatch"


async def _attach_pubmed_abstract_if_empty(
    document: DocumentEnvelope,
    payload: Mapping[str, Any],
    fetch_pubmed: Callable[[str], Any] | None,
) -> DocumentEnvelope:
    if document.chunks:
        return document
    doi = next(
        (item.value for item in document.identifiers if item.scheme == "doi"),
        None,
    )
    ids = payload.get("ids")
    pmid = _numeric_pmid(ids.get("pmid")) if isinstance(ids, Mapping) else None
    status: Dict[str, Any] = {
        "status": "missing_doi" if not doi else "missing_pmid"
    }
    document.metadata["pubmed_abstract"] = status
    if not doi or not pmid:
        return document
    status.update(pmid=pmid, doi=doi)
    if fetch_pubmed is None:
        from .connectors import fetch_pubmed_xml as fetch_pubmed
    try:
        raw = await resolve(fetch_pubmed(pmid))
    except Exception as exc:
        logger.warning(
            "PubMed PMID %s fetch failed: %s", pmid, type(exc).__name__
        )
        status.update(status="fetch_failed", error=type(exc).__name__)
        return document
    abstract, reason = _pubmed_abstract(raw, pmid, doi)
    if abstract is None:
        status["status"] = reason
        return document
    try:
        snapshot = snapshot_bytes(raw)
    except Exception as exc:
        logger.warning(
            "PubMed PMID %s snapshot failed: %s", pmid, type(exc).__name__
        )
        status.update(status="snapshot_failed", error=type(exc).__name__)
        return document
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    document.chunks.append(
        Chunk(
            chunk_id=stable_id(
                "chunk", document.document_version_id, "abstract", 0, abstract
            ),
            kind="abstract",
            text=abstract,
            order=0,
            section_path=["abstract"],
            locator={
                "source": "pubmed",
                "pmid": pmid,
                "doi": doi,
                "url": pubmed_url,
                "xpath": "PubmedArticle/MedlineCitation/Article/Abstract",
                "snapshot_uri": snapshot.as_uri(),
                "snapshot_sha256": snapshot.name,
            },
        )
    )
    document.coverage = "abstract_only"
    status.update(
        status="parsed",
        url=pubmed_url,
        snapshot_uri=snapshot.as_uri(),
        sha256=snapshot.name,
        byte_length=len(raw),
    )
    logger.debug(
        "%s: PubMed abstract attached from PMID %s",
        document.document_version_id,
        pmid,
    )
    return document


def select_sections(chunks: List[Chunk]) -> tuple:
    """The PDF chunks worth the model's calls: method, conclusion,
    introduction and similar sections by role priority, up to max_chunks,
    in document order; the head and the tail when no heading is known.
    Returns the kept chunks and a summary for the audit.
    """
    settings = load_catalog("pipeline")["openalex_fulltext"].get("sections")
    limit = int((settings or {}).get("max_chunks", 0))
    if not settings or not limit or len(chunks) <= limit:
        return chunks, {"total": len(chunks), "kept": len(chunks)}
    roles = list(settings["roles"])
    rank = {role: index for index, role in enumerate(roles)}
    by_role = [
        (rank[role], index)
        for index, chunk in enumerate(chunks)
        if (role := chunk.locator.get("section_role")) in rank
    ]
    if by_role:
        chosen = sorted(index for _, index in sorted(by_role)[:limit])
        basis = "sections"
    else:
        head = int(settings["head_chunks"])
        tail = int(settings["tail_chunks"])
        chosen = sorted(
            {
                *range(min(head, len(chunks))),
                *range(max(0, len(chunks) - tail), len(chunks)),
            }
        )[:limit]
        basis = "head_and_tail"
    kept = [chunks[index] for index in chosen]
    return kept, {
        "total": len(chunks),
        "kept": len(kept),
        "basis": basis,
        "roles": sorted(
            {chunk.locator.get("section_role") for chunk in kept} - {None}
        ),
    }


async def attach_openalex_fulltext(
    document: DocumentEnvelope,
    payload: Mapping[str, Any],
    fetch: Callable[[str], Any] = None,
    fetch_pubmed: Callable[[str], Any] | None = None,
    known_sha256: Collection[str] = (),
) -> DocumentEnvelope:
    """Attach an open-access PDF, then fall back to a verified PubMed abstract.

    ``fetch`` and ``fetch_pubmed`` may be sync or async. Docling runs in a
    worker thread with a timeout, so a pathological PDF fails this document
    instead of the job. A PDF whose bytes are in ``known_sha256`` was already
    extracted: it is recorded as ``already_processed`` and not parsed again.
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
        return await _attach_pubmed_abstract_if_empty(
            document, payload, fetch_pubmed
        )
    require_pdf_support()
    for url in urls:
        try:
            raw = await resolve(fetch(url))
            sha256 = hashlib.sha256(raw).hexdigest()
            if sha256 in known_sha256:
                status.update(
                    status="already_processed",
                    pdf_url=url,
                    sha256=sha256,
                    byte_length=len(raw),
                )
                logger.debug("Full text %s was already extracted", url)
                return document
            # The converter enforces pdf_timeout_seconds itself and kills a
            # stuck conversion; a thread timeout here would leave it running.
            snapshot, chunks, warnings = await asyncio.to_thread(
                _body_chunks, raw, document, url
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
        chunks, selection = select_sections(chunks)
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
            selection=selection,
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
    return await _attach_pubmed_abstract_if_empty(
        document, payload, fetch_pubmed
    )
