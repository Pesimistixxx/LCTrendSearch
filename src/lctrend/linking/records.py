"""Grants and vacancies linked to technologies the graph already knows.

A grant or a vacancy carries its money in structured fields (the economic
parsers read it); the model is only needed to learn which technology the
record is about. When the text names a technology of the registry by one
of its reviewed names (linking.names), that link is made here without a
model call, and the record goes straight to the graph. A record that names
no known technology returns None and takes the LLM path, which can
discover new ones.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from ..core.models import (
    Concept,
    DocumentEnvelope,
    DocumentType,
    ExtractionResult,
    Mention,
    ProcessingRun,
    ResolutionDecision,
    json_value,
    stable_id,
)
from ..extraction.economics import extract_economic_evidence
from ..ingest.processed import fulltext_sha256
from .names import MIN_LETTERS, NameIndex

PARSER = "registry_match"
METHOD = "registry_lexical_match"
# Records whose model-free link is enough: the money is structured, the
# text is short and names its technologies as skills or keywords.
ECONOMIC_TYPES = frozenset({DocumentType.GRANT, DocumentType.JOB_POSTING})


def link_known_technologies(
    document: DocumentEnvelope, registry: Iterable[Concept]
) -> Optional[ExtractionResult]:
    """A model-free extraction of the known technologies the record names,
    or None when it names none (the record then needs the model)."""
    index = NameIndex(registry, families={"technology"})
    if not index or not document.chunks:
        return None
    mentions: List[Mention] = []
    decisions: List[ResolutionDecision] = []
    linked: Dict[str, Concept] = {}
    for chunk in document.chunks:
        # One mention of a technology per chunk: repeating a skill in a
        # list is not more evidence.
        seen = set()
        for concept_id, start, end in index.find(chunk.text):
            if concept_id in seen:
                continue
            seen.add(concept_id)
            concept = index.concepts[concept_id]
            surface = chunk.text[start:end]
            mention_id = stable_id(
                "mention",
                document.document_version_id,
                json_value([chunk.chunk_id, start, end]),
                concept.preferred_label,
                concept.kind.value,
            )
            mentions.append(
                Mention(
                    mention_id=mention_id,
                    chunk_id=chunk.chunk_id,
                    surface_text=surface,
                    canonical_text=concept.preferred_label,
                    start=start,
                    end=end,
                    type_candidates=[concept.kind],
                    mention_role="entity",
                )
            )
            decisions.append(
                ResolutionDecision(
                    resolution_id=stable_id("resolution", mention_id, METHOD),
                    mention_id=mention_id,
                    status="accepted",
                    concept_id=concept_id,
                    method=METHOD,
                    score=1.0,
                    basis=["reviewed name of a known concept in the text"],
                    review_status="not_required",
                )
            )
            linked.setdefault(concept_id, concept.model_copy(deep=True))
    if not mentions:
        return None
    concepts = list(linked.values())
    run = ProcessingRun(
        run_id=stable_id("run", document.document_version_id, uuid4()),
        pipeline_version="registry-match/1",
        parser=PARSER,
        started_at=datetime.now(timezone.utc).isoformat(),
        config_hash=stable_id("config", PARSER, MIN_LETTERS),
        metadata={
            "extraction": "registry_match",
            "model_calls": 0,
            "issues": [],
            "coverage": {
                "total_chunks": len(document.chunks),
                "processed_focus_chunk_ids": sorted(
                    chunk.chunk_id for chunk in document.chunks
                ),
                "unprocessed_chunk_ids": [],
            },
            "input_coverage": document.coverage,
            "input_fulltext_sha256": fulltext_sha256(document),
            "linked_concepts": sorted(linked),
        },
    )
    return ExtractionResult(
        document_version_id=document.document_version_id,
        run=run,
        mentions=mentions,
        concepts=concepts,
        resolutions=decisions,
        economic_evidence=extract_economic_evidence(
            document.chunks, mentions, concepts, decisions
        ),
    )
