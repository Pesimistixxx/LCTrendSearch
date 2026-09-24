from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

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
    return mentions
