"""Registry concepts a packet names or resembles, as reference names.

The extractor reads a packet blind to the graph, so one technology can
enter it under several local names ("GNN-based fraud detection", "graph
neural networks for fraud detection") and split its signal across
concepts. Before the call, the packet's text is matched against the
registry:

- ``name``: a reviewed name of the concept occurs in the text
  (linking.names, the resolver's identity key);
- ``similar``: a chunk of the packet is close to the concept's cached
  vector (resolver.concept_text, "label: definition"); only vectors
  already embedded or seeded from the graph are compared, so this costs
  one embedding request per packet and none for the registry.

The block is a reference, never evidence: validation still requires every
label to be written in the source (llm.validation), so the model can pick
the known form of a name the text uses but cannot rename a new technology
into a known one.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import load_catalog
from ..core.models import Concept
from ..extraction.resolver import concept_text
from .names import INACTIVE, NameIndex

logger = logging.getLogger(__name__)

_CACHE: Dict[Tuple, "KnownConcepts"] = {}
_CACHE_LOCK = Lock()


def settings() -> Dict[str, Any]:
    return dict(
        load_catalog("pipeline").get("linking", {}).get("known_concepts", {})
    )


class KnownConcepts:
    """Name index and vector matrix of one registry state."""

    def __init__(
        self,
        registry: Sequence[Concept],
        semantic: Any = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.options = options if options is not None else settings()
        kinds = set(self.options.get("kinds") or [])
        self.concepts = [
            concept
            for concept in registry
            if concept.status not in INACTIVE
            and (not kinds or concept.kind.value in kinds)
        ]
        self.by_id = {concept.concept_id: concept for concept in self.concepts}
        self.names = NameIndex(self.concepts)
        self.semantic = (
            semantic
            if self.options.get("semantic", True)
            and getattr(semantic, "cached_vector", None) is not None
            else None
        )
        self.vector_ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None
        if self.semantic is not None:
            rows = []
            for concept in self.concepts:
                vector = self.semantic.cached_vector(concept_text(concept))
                if vector:
                    self.vector_ids.append(concept.concept_id)
                    rows.append(vector)
            if rows:
                self.matrix = np.asarray(rows, dtype=np.float32)

    @classmethod
    def for_registry(
        cls, registry: Any, semantic: Any = None
    ) -> "KnownConcepts":
        """Shared by the documents of a job while the registry keeps its
        size; a grown registry (new concepts) is indexed again."""
        concepts = list(registry)
        key = (
            id(registry),
            len(concepts),
            id(semantic),
            getattr(semantic, "embedding_model_name", None),
        )
        with _CACHE_LOCK:
            found = _CACHE.get(key)
        if found is not None:
            return found
        built = cls(concepts, semantic)
        with _CACHE_LOCK:
            _CACHE.clear()
            _CACHE[key] = built
        return built

    def __bool__(self) -> bool:
        return bool(self.concepts)

    def _similar(self, texts: List[str]) -> List[Tuple[str, float]]:
        if self.matrix is None or not texts:
            return []
        limit = int(self.options.get("max_chunk_chars", 2000))
        vectors = self.semantic.embed(
            [text[:limit] for text in texts], cache=False
        )
        if not vectors:
            return []
        scores = (self.matrix @ np.asarray(vectors, dtype=np.float32).T).max(
            axis=1
        )
        floor = float(self.options.get("semantic_min_cosine", 0.55))
        order = np.argsort(-scores)
        return [
            (self.vector_ids[index], float(scores[index]))
            for index in order[: int(self.options.get("limit", 15))]
            if scores[index] >= floor
        ]

    def entry(self, concept: Concept, match: str, score=None) -> dict:
        names = [
            name.text
            for name in concept.names
            if name.status == "accepted"
            and name.text.casefold() != concept.preferred_label.casefold()
        ][: int(self.options.get("max_names", 3))]
        entry = {
            "label": concept.preferred_label,
            "kind": concept.kind.value,
            "match": match,
        }
        if names:
            entry["names"] = names
        if concept.definition:
            limit = int(self.options.get("max_definition_chars", 160))
            entry["definition"] = concept.definition[:limit]
        if score is not None:
            entry["cosine"] = round(score, 3)
        return entry

    def lookup(self, texts: List[str]) -> List[dict]:
        """Named concepts first (in text order), then the most similar.

        Synchronous (the semantic layer makes a blocking request): call it
        from a worker thread.
        """
        limit = int(self.options.get("limit", 15))
        chosen: Dict[str, dict] = {}
        for text in texts:
            for concept_id, _, _ in self.names.find(text):
                if concept_id not in chosen and len(chosen) < limit:
                    chosen[concept_id] = self.entry(
                        self.by_id[concept_id], "name"
                    )
        if len(chosen) < limit:
            try:
                similar = self._similar(texts)
            except Exception as exc:
                # A reference block is optional; the packet goes without it.
                logger.warning(
                    "Known concepts by similarity skipped (%s)",
                    type(exc).__name__,
                )
                similar = []
            for concept_id, score in similar:
                if concept_id not in chosen and len(chosen) < limit:
                    chosen[concept_id] = self.entry(
                        self.by_id[concept_id], "similar", score
                    )
        return list(chosen.values())


def enabled() -> bool:
    return bool(settings().get("enabled", False))
