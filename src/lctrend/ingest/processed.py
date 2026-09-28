"""Whether an earlier extraction already covers a document's input.

A version identifies the source record, not the text the model read: the
same OpenAlex work may first be processed from its abstract and later from
an open-access PDF. A prior run covers the document only when it read at
least as much text, and for a full text, the same PDF bytes.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..core.aio import call
from ..core.models import DocumentEnvelope

# How much source text a coverage level gives the model.
COVERAGE_RANK = {
    "metadata_only": 0,
    "abstract_only": 1,
    "selected_files": 1,
    "selected_text": 1,
    "parsed_text": 1,
    "full_text": 2,
    "abstract_and_full_text": 2,
}
# Runs written before the input was recorded were full extractions of
# whatever was available; they are not paid for again.
LEGACY_RANK = max(COVERAGE_RANK.values())


def run_input(metadata: Any) -> Dict[str, Any]:
    """The input a stored run read, from its metadata (dict or JSON)."""
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except ValueError:
            metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "coverage": metadata.get("input_coverage"),
        "fulltext_sha256": metadata.get("input_fulltext_sha256"),
    }


def fulltext_sha256(document: DocumentEnvelope) -> Optional[str]:
    """SHA-256 of the PDF whose text the document carries (or skipped)."""
    status = document.metadata.get("fulltext")
    if isinstance(status, Mapping) and status.get("status") in {
        "parsed",
        "already_processed",
    }:
        return status.get("sha256")
    return None


def _rank(coverage: Any) -> int:
    if coverage is None:
        return LEGACY_RANK
    return COVERAGE_RANK.get(str(coverage), 1)


def covers(
    prior: Iterable[Mapping[str, Any]], document: DocumentEnvelope
) -> bool:
    """A prior run read at least this document's input."""
    sha = fulltext_sha256(document)
    needed = _rank(document.coverage)
    for item in prior:
        if sha is not None:
            if item.get("fulltext_sha256") == sha:
                return True
            # A legacy full-text run did not record its PDF.
            if (
                item.get("fulltext_sha256") is None
                and _rank(item.get("coverage")) >= COVERAGE_RANK["full_text"]
            ):
                return True
            continue
        if _rank(item.get("coverage")) >= needed:
            return True
    return False


def known_fulltexts(prior: Iterable[Mapping[str, Any]]) -> List[str]:
    """PDFs already read: a download with these bytes needs no parsing."""
    return [
        item["fulltext_sha256"]
        for item in prior
        if item.get("fulltext_sha256")
    ]


async def prior_inputs(store: Any, version_id: str) -> List[Dict[str, Any]]:
    """Inputs of complete extractions of this version (empty if unknown).

    Stores without input tracking report only whether a version was
    processed; such a run counts as a legacy full extraction.
    """
    lookup = getattr(store, "processed_inputs", None)
    if lookup is not None:
        found = await call(lookup, [version_id])
        return list(found.get(version_id, []))
    lookup = getattr(store, "processed_versions", None)
    if lookup is None:
        return []
    found = await call(lookup, [version_id])
    return [run_input({})] if version_id in found else []
