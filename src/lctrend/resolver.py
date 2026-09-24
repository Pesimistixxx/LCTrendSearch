from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

from .models import (
    Concept,
    ConceptKind,
    ConceptName,
    Mention,
    ResolutionDecision,
    stable_id,
)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", value)


def resolve_exact_mentions(
    mentions: Sequence[Mention], registry: Iterable[Concept]
) -> Tuple[List[Concept], List[ResolutionDecision]]:
    """Resolve reviewed exact aliases; preserve everything else as provisional."""
    index: DefaultDict[str, List[Concept]] = defaultdict(list)
    for concept in registry:
        accepted_names = [name for name in concept.names if name.status == "accepted"]
        if concept.status == "accepted" and not accepted_names:
            accepted_names = [
                ConceptName(
                    name_id=stable_id("name", concept.concept_id, concept.preferred_label),
                    text=concept.preferred_label,
                    normalized_text=normalize_name(concept.preferred_label),
                )
            ]
        for name in accepted_names:
            index[name.normalized_text].append(concept)

    new_concepts: List[Concept] = []
    decisions: List[ResolutionDecision] = []
    for mention in mentions:
        normalized = normalize_name(mention.surface_text)
        compatible = [
            concept
            for concept in index.get(normalized, [])
            if concept.kind in mention.type_candidates or ConceptKind.CANDIDATE in mention.type_candidates
        ]
        resolution_id = stable_id("resolution", mention.mention_id, "exact-v1")
        if len(compatible) == 1:
            decisions.append(
                ResolutionDecision(
                    resolution_id=resolution_id,
                    mention_id=mention.mention_id,
                    status="accepted",
                    concept_id=compatible[0].concept_id,
                    method="reviewed_exact_name",
                    score=1.0,
                    basis=["normalized name equals a reviewed name", "concept type is compatible"],
                    review_status="not_required",
                )
            )
            continue

        if len(compatible) > 1:
            decisions.append(
                ResolutionDecision(
                    resolution_id=resolution_id,
                    mention_id=mention.mention_id,
                    status="ambiguous",
                    candidates=[{"concept_id": item.concept_id, "score": 1.0} for item in compatible],
                    method="reviewed_exact_name",
                    score=1.0,
                    basis=["same reviewed name maps to multiple compatible concepts"],
                )
            )
            continue

        kind = mention.type_candidates[0] if mention.type_candidates else ConceptKind.CANDIDATE
        concept_id = stable_id("concept", "provisional", mention.mention_id)
        new_concepts.append(
            Concept(
                concept_id=concept_id,
                kind=kind,
                preferred_label=mention.surface_text,
                status="provisional",
                names=[
                    ConceptName(
                        name_id=stable_id("name", concept_id, normalized),
                        text=mention.surface_text,
                        normalized_text=normalized,
                        name_kind="observed",
                        status="provisional",
                    )
                ],
            )
        )
        decisions.append(
            ResolutionDecision(
                resolution_id=resolution_id,
                mention_id=mention.mention_id,
                status="provisional",
                concept_id=concept_id,
                method="new_provisional",
                basis=["no reviewed exact name of a compatible type"],
            )
        )

    return new_concepts, decisions
