"""Chunks the model does not need, skipped before any call.

A paper's description of its data and sample ("We surveyed senior human
resources managers in three waves...") or its administrative statements
(author contributions, competing interests) name no technology the radar
tracks, yet each costs an extraction call and a review call. Their role
comes from the section heading (``locator.section_role``, set by the PDF
and file parsers from pipeline.json ``section_roles``) or, for a chunk
without one, from a heading that opens the chunk ("A. Data Source: ...").

Acknowledgments are kept: they name the funders of the work.
A document made only of skipped chunks is read whole, so a data paper is
never left without extraction.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

from ..core.config import load_catalog
from ..core.models import Chunk, DocumentEnvelope

SKIPPED_REASON = "skipped_section:"
# A heading opening a chunk is at its start.
_OPENING_CHARS = 120


@lru_cache(maxsize=1)
def _rules() -> Tuple[Tuple[str, "re.Pattern[str]"], ...]:
    catalog = load_catalog("pipeline")
    skipped = catalog.get("linking", {}).get("skipped_section_roles", [])
    roles = catalog["section_roles"]
    return tuple(
        (role, re.compile(roles[role], re.IGNORECASE))
        for role in skipped
        if role in roles
    )


def skipped_role(chunk: Chunk) -> Optional[str]:
    """The skipped section role of a chunk, or None when it is read."""
    rules = _rules()
    if not rules:
        return None
    role = chunk.locator.get("section_role")
    if role is not None:
        return role if role in dict(rules) else None
    # The patterns are anchored at the heading's start and end at its colon
    # or line end, so "A. Data Source: Workplace organization ..." opening
    # a paragraph matches and "Data were collected ..." does not.
    opening = chunk.text.lstrip()[:_OPENING_CHARS].split("\n", 1)[0]
    return next(
        (name for name, pattern in rules if pattern.search(opening)), None
    )


def skipped_chunks(document: DocumentEnvelope) -> Dict[str, str]:
    """chunk_id -> skipped role; empty when nothing else would be read."""
    found = {}
    readable = 0
    for chunk in document.chunks:
        if not chunk.text.strip() or chunk.parse_status == "rejected":
            continue
        readable += 1
        role = skipped_role(chunk)
        if role is not None:
            found[chunk.chunk_id] = role
    return found if len(found) < readable else {}
