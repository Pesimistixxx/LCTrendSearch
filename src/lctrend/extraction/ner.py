from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..core.config import load_catalog
from ..core.models import ConceptKind, DocumentEnvelope, Mention, stable_id


def default_labels() -> Dict[str, ConceptKind]:
    return {
        label: ConceptKind(kind)
        for label, kind in load_catalog("extraction")["ner"]["labels"].items()
    }


# Compatibility export. Extraction reads the active catalog again at call time.
DEFAULT_LABELS = default_labels()


def _composite_technologies(chunk_id: str, text: str) -> List[Mention]:
    composites = []
    settings = load_catalog("extraction")
    if not settings["ner"]["composite_technologies_enabled"]:
        return composites
    for sentence in re.finditer(settings["sentence_pattern"], text):
        sentence_text = sentence.group()
        for rule in settings["ner"]["composite_technology_rules"]:
            method, application, canonical = (
                rule["method"],
                rule["application"],
                rule["canonical"],
            )
            method_match = re.search(method, sentence_text, re.IGNORECASE)
            application_match = re.search(
                application, sentence_text, re.IGNORECASE
            )
            if not method_match or not application_match:
                continue
            start = sentence.start() + method_match.start()
            end = sentence.start() + application_match.end()
            if end <= start:
                continue
            composites.append(
                Mention(
                    mention_id=stable_id(
                        "mention", chunk_id, start, end, "composite_technology"
                    ),
                    chunk_id=chunk_id,
                    surface_text=text[start:end],
                    canonical_text=canonical,
                    start=start,
                    end=end,
                    type_candidates=[ConceptKind.TECHNOLOGY],
                    confidence=None,
                )
            )
    return composites


def extract_mentions(
    document: DocumentEnvelope,
    model: Any,
    labels: Optional[Mapping[str, ConceptKind]] = None,
    threshold: Optional[float] = None,
) -> List[Mention]:
    settings = load_catalog("extraction")["ner"]
    labels = default_labels() if labels is None else labels
    threshold = settings["threshold"] if threshold is None else threshold
    generic_technologies = {
        name.casefold() for name in settings["generic_technologies"]
    }
    mentions: List[Mention] = []
    for chunk in document.chunks:
        entities: Iterable[Dict[str, Any]] = model.predict_entities(
            chunk.text, list(labels), threshold=threshold
        )
        for entity in entities:
            label = str(entity["label"])
            start, end = int(entity["start"]), int(entity["end"])
            surface = chunk.text[start:end]
            mentions.append(
                Mention(
                    mention_id=stable_id(
                        "mention", chunk.chunk_id, start, end, label
                    ),
                    chunk_id=chunk.chunk_id,
                    surface_text=surface,
                    start=start,
                    end=end,
                    type_candidates=[labels.get(label, ConceptKind.CANDIDATE)],
                    confidence=entity.get("score"),
                )
            )
        composites = _composite_technologies(chunk.chunk_id, chunk.text)
        if composites:
            composite_ranges = [(item.start, item.end) for item in composites]
            mentions = [
                item
                for item in mentions
                if not (
                    item.chunk_id == chunk.chunk_id
                    and item.type_candidates == [ConceptKind.TECHNOLOGY]
                    and item.surface_text.casefold() in generic_technologies
                    and any(
                        start <= item.start <= end
                        for start, end in composite_ranges
                    )
                )
            ]
            mentions.extend(composites)
    # Broad fields from auxiliary NER are still preserved for audit, but do
    # not establish a technology identity without contextual LLM review.
    for mention in mentions:
        if (
            mention.type_candidates == [ConceptKind.TECHNOLOGY]
            and mention.surface_text.casefold() in generic_technologies
        ):
            mention.type_candidates = [ConceptKind.CANDIDATE]
    mentions.sort(
        key=lambda item: (item.chunk_id, item.start, item.end, item.mention_id)
    )
    return mentions
