"""Conservative local candidates, with literal evidence and no acceptance
shortcut.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from ..core.config import load_catalog
from ..core.models import (
    Assertion,
    Chunk,
    Concept,
    EvidenceSpan,
    Mention,
    ResolutionDecision,
    stable_id,
)


def claim_context(text: str, settings: dict | None = None) -> tuple[str, str]:
    """Preserve negation and intent instead of turning every relation into a
    fact.
    """
    rules = (settings or load_catalog("extraction"))["polarity"]
    polarity = (
        "negated"
        if re.search(rules["negated"], text, re.IGNORECASE)
        else "affirmed"
    )
    if re.search(rules["hypothetical"], text, re.IGNORECASE):
        modality = "hypothetical"
    elif re.search(rules["planned"], text, re.IGNORECASE):
        modality = "planned"
    else:
        modality = "reported"
    return polarity, modality


def resolved_ids(resolutions: Sequence[ResolutionDecision]) -> dict[str, str]:
    # An observed concept can support a review candidate; an ambiguous
    # mapping cannot.
    return {
        item.mention_id: item.concept_id
        for item in resolutions
        if item.concept_id and item.status in {"accepted", "provisional"}
    }


def extract_assertions(
    chunks: Iterable[Chunk],
    mentions: Sequence[Mention],
    concepts: Sequence[Concept],
    resolutions: Sequence[ResolutionDecision],
) -> List[Assertion]:
    """Extract explicit subject–verb–object candidates with unambiguous roles.

    This intentionally sacrifices recall. Co-occurrence, 'for', 'to' and 'with'
    cannot prove a technology solves a task. Reviewed imports produce
    acceptance.
    """
    settings = load_catalog("extraction")
    rules = settings["assertions"]
    kinds = {concept.concept_id: concept.kind.value for concept in concepts}
    decisions = resolved_ids(resolutions)
    by_chunk: dict[str, list[Mention]] = {}
    for mention in mentions:
        if decisions.get(mention.mention_id) in kinds:
            by_chunk.setdefault(mention.chunk_id, []).append(mention)
    output: list[Assertion] = []
    seen = set()
    for chunk in chunks:
        for sentence in re.finditer(settings["sentence_pattern"], chunk.text):
            raw = sentence.group()
            quote = raw.strip()
            if not quote:
                continue
            start = sentence.start() + len(raw) - len(raw.lstrip())
            end = start + len(quote)
            local = [
                item
                for item in by_chunk.get(chunk.chunk_id, [])
                if start <= item.start < item.end <= end
            ]
            subjects = [
                item
                for item in local
                if kinds[decisions[item.mention_id]] in rules["subject_kinds"]
            ]
            if len({decisions[item.mention_id] for item in subjects}) != 1:
                continue
            for rule in rules["relation_rules"]:
                objects = [
                    item
                    for item in local
                    if kinds[decisions[item.mention_id]]
                    in rules["object_kinds"][rule["predicate"]]
                ]
                if len({decisions[item.mention_id] for item in objects}) != 1:
                    continue
                for subject in subjects:
                    # Strict subject position prevents
                    # passive/comparison/attribution reversal.
                    if chunk.text[start : subject.start].strip():
                        continue
                    for target in objects:
                        if subject.end >= target.start:
                            continue
                        connector = chunk.text[subject.end : target.start]
                        if not re.fullmatch(
                            rule["pattern"], connector, re.IGNORECASE
                        ):
                            continue
                        roles = {
                            "subject": decisions[subject.mention_id],
                            rule["role"]: decisions[target.mention_id],
                        }
                        polarity, modality = claim_context(connector, settings)
                        key = (
                            chunk.chunk_id,
                            start,
                            end,
                            rule["predicate"],
                            tuple(sorted(roles.items())),
                            polarity,
                            modality,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        output.append(
                            Assertion(
                                assertion_id=stable_id("assertion", *key),
                                predicate=rule["predicate"],
                                roles=roles,
                                evidence=[
                                    EvidenceSpan(
                                        chunk_id=chunk.chunk_id,
                                        quote=quote,
                                        start=start,
                                        end=end,
                                        supports_fields=[
                                            "predicate",
                                            "roles",
                                            "polarity",
                                            "modality",
                                        ],
                                    )
                                ],
                                qualifiers={
                                    "extraction_method": (
                                        "explicit_relation_rule"
                                    ),
                                    "relation_text": connector.strip(),
                                },
                                polarity=polarity,
                                modality=modality,
                                extraction_confidence=None,
                                verification_status="unverified",
                                status="needs_review",
                            )
                        )
    return output
