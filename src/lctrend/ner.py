from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .models import ConceptKind, DocumentEnvelope, Mention, stable_id


DEFAULT_LABELS: Dict[str, ConceptKind] = {
    "technology": ConceptKind.TECHNOLOGY,
    "method": ConceptKind.METHOD,
    "task": ConceptKind.TASK,
    "technical problem": ConceptKind.PROBLEM,
    "application": ConceptKind.APPLICATION_CONTEXT,
    "metric": ConceptKind.METRIC,
    "material": ConceptKind.MATERIAL,
}

COMPOSITE_TECHNOLOGY_RULES: Tuple[Tuple[str, str, str], ...] = (
    (r"(?:vision )?transformers?|transformer[- ]based(?: models?)?", r"computer vision|image (?:analysis|classification|segmentation)", "Transformer-based computer vision"),
    (r"transformers?|transformer[- ]based(?: models?)?", r"natural language processing|text classification|named entity recognition", "Transformer-based natural language processing"),
)
GENERIC_TECHNOLOGIES = {"ai", "artificial intelligence", "computer vision", "cv", "machine learning", "natural language processing", "nlp"}


def _composite_technologies(chunk_id: str, text: str) -> List[Mention]:
    composites = []
    for sentence in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text):
        sentence_text = sentence.group()
        for method, application, canonical in COMPOSITE_TECHNOLOGY_RULES:
            method_match = re.search(method, sentence_text, re.IGNORECASE)
            application_match = re.search(application, sentence_text, re.IGNORECASE)
            if not method_match or not application_match:
                continue
            start = sentence.start() + method_match.start()
            end = sentence.start() + application_match.end()
            if end <= start:
                continue
            composites.append(
                Mention(
                    mention_id=stable_id("mention", chunk_id, start, end, "composite_technology"),
                    chunk_id=chunk_id,
                    surface_text=text[start:end],
                    canonical_text=canonical,
                    start=start,
                    end=end,
                    type_candidates=[ConceptKind.TECHNOLOGY],
                    confidence=1.0,
                )
            )
    return composites


def extract_mentions(
    document: DocumentEnvelope,
    model: Any,
    labels: Mapping[str, ConceptKind] = DEFAULT_LABELS,
    threshold: float = 0.5,
) -> List[Mention]:
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
                    mention_id=stable_id("mention", chunk.chunk_id, start, end, label),
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
                item for item in mentions
                if not (
                    item.chunk_id == chunk.chunk_id
                    and item.type_candidates == [ConceptKind.TECHNOLOGY]
                    and item.surface_text.casefold() in GENERIC_TECHNOLOGIES
                    and any(start <= item.start <= end for start, end in composite_ranges)
                )
            ]
            mentions.extend(composites)
    mentions.sort(key=lambda item: (item.chunk_id, item.start, item.end, item.mention_id))
    return mentions
