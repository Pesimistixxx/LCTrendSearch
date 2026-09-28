from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from time import monotonic
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.config import load_catalog
from ..core.models import (
    SEMANTIC_CANDIDATE_METHOD,
    Concept,
    ConceptKind,
    ConceptName,
    Mention,
    ResolutionDecision,
    stable_id,
)
from .lexical import identity_key, lexical_key


# Names repeat across documents; the caches keep resolution linear in the
# number of mentions instead of re-lemmatizing the whole registry each time.
@lru_cache(maxsize=262144)
def normalize_name(value: str) -> str:
    value = "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char
        for char in value
    )
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def lemmatize_name(value: str) -> str:
    return lexical_key(value)


def _alias_keys(value: str, kind: object = None) -> frozenset[str]:
    # Initials are retrieval hints, never identity evidence (CC has many
    # meanings).
    return frozenset({f"key:{identity_key(value, kind)}"})


def alias_keys(value: str, kind: object = None) -> set[str]:
    return set(_alias_keys(value, kind))


EMBEDDING_PROVIDERS = ("gigachat", "transformers")
# After a failure (no credentials, no torch, HTTP 5xx) resolution continues
# lexically; the semantic layer is retried after this pause.
SEMANTIC_RETRY_SECONDS = 300.0

logger = logging.getLogger(__name__)


def _unit(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


class SemanticDeduplicator:
    """Retrieve review candidates; semantic similarity does not establish
    identity.

    Embeddings come from the GigaChat /embeddings API by default or from a
    local transformers model; the pair check stays a local cross-encoder.
    """

    def __init__(
        self,
        cosine_threshold: Optional[float] = None,
        decision_threshold: Optional[float] = None,
        embedding_model: Optional[str] = None,
        decision_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedder=None,
    ) -> None:
        settings = load_catalog("resolver")["semantic"]
        self.embedding_provider = (
            (embedding_provider or settings["embedding_provider"])
            .strip()
            .lower()
        )
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            raise ValueError(
                "embedding provider must be one of "
                + ", ".join(EMBEDDING_PROVIDERS)
            )
        self.cosine_threshold = (
            settings["cosine_threshold"]
            if cosine_threshold is None
            else cosine_threshold
        )
        self.decision_threshold = (
            settings["decision_threshold"]
            if decision_threshold is None
            else decision_threshold
        )
        self.embedding_model_name = (
            embedding_model
            or settings["embedding_models"][self.embedding_provider]
        )
        self.decision_model_name = decision_model or settings["decision_model"]
        self.max_length = settings["max_length"]
        self.batch_size = settings["embedding_batch_size"]
        self._embedder = embedder
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._decision_tokenizer = None
        self._decision_model = None
        self._cache: Dict[str, List[float]] = {}
        self.failure: Optional[str] = None
        self._failed_at = 0.0

    def available(self) -> bool:
        return (
            self.failure is None
            or monotonic() - self._failed_at >= SEMANTIC_RETRY_SECONDS
        )

    def _fail(self, exc: Exception) -> None:
        self.failure = type(exc).__name__
        self._failed_at = monotonic()
        logger.warning(
            "Semantic layer unavailable (%s); resolving lexically for %.0fs",
            self.failure,
            SEMANTIC_RETRY_SECONDS,
        )

    def embed(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        """Unit vectors of labels, or None when the layer is unavailable.

        Synchronous: call it from a worker thread, as resolution does.
        """
        if not self.available():
            return None
        try:
            vectors = self._embed(texts)
        except Exception as exc:
            self._fail(exc)
            return None
        self.failure = None
        return vectors

    def _local_embeddings(self, texts: List[str]) -> List[List[float]]:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._embedding_model is None:
            self._embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_name
            )
            self._embedding_model = AutoModel.from_pretrained(
                self.embedding_model_name
            )
            self._embedding_model.eval()
        encoded = self._embedding_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            output = self._embedding_model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        vectors = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        return vectors.tolist()

    def _remote_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self._embedder is None:
            from ..llm.client import JsonLLM

            # Reuses GIGACHAT_CREDENTIALS, scope, base URL and CA bundle.
            self._embedder = JsonLLM(provider="gigachat")
        from ..core.aio import resolve, run_sync

        # Resolution runs in a worker thread, off the event loop.
        return run_sync(
            resolve(self._embedder.embed(texts, self.embedding_model_name))
        )

    def _embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Unit vectors for texts; one batched request per uncached group."""
        texts = list(texts)
        keys = [normalize_name(text) for text in texts]
        missing = list(
            dict.fromkeys(
                text
                for text, key in zip(texts, keys)
                if key not in self._cache
            )
        )
        compute = (
            self._remote_embeddings
            if self.embedding_provider == "gigachat"
            else self._local_embeddings
        )
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            for text, vector in zip(batch, compute(batch)):
                self._cache.setdefault(normalize_name(text), _unit(vector))
        return [self._cache[key] for key in keys]

    def _decision_score(self, left: str, right: str) -> float:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if self._decision_model is None:
            self._decision_tokenizer = AutoTokenizer.from_pretrained(
                self.decision_model_name
            )
            self._decision_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.decision_model_name
                )
            )
            self._decision_model.eval()
        encoded = self._decision_tokenizer(
            left,
            right,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.no_grad():
            logits = self._decision_model(**encoded).logits.squeeze()
        if logits.numel() == 1:
            return float(torch.sigmoid(logits).item())
        return float(torch.softmax(logits, dim=-1)[-1].item())

    def best_match(
        self, text: str, concepts: Sequence[Concept]
    ) -> Tuple[Optional[Concept], float, float]:
        if not concepts or not self.available():
            return None, 0.0, 0.0
        try:
            result = self._best_match(text, concepts)
        except Exception as exc:
            self._fail(exc)
            return None, 0.0, 0.0
        self.failure = None
        return result

    def _best_match(
        self, text: str, concepts: Sequence[Concept]
    ) -> Tuple[Optional[Concept], float, float]:
        source, *targets = self._embed(
            [text, *(concept.preferred_label for concept in concepts)]
        )
        scored = [
            (sum(a * b for a, b in zip(source, target)), concept)
            for target, concept in zip(targets, concepts)
        ]
        cosine, concept = max(scored, key=lambda item: item[0])
        if cosine < self.cosine_threshold:
            return None, cosine, 0.0
        decision = self._decision_score(text, concept.preferred_label)
        return (
            (concept if decision >= self.decision_threshold else None),
            cosine,
            decision,
        )


def _aliases(concept: Concept) -> List[str]:
    # An unreviewed observed name must not become identity evidence for the
    # next document.
    return [
        concept.preferred_label,
        *(name.text for name in concept.names if name.status == "accepted"),
    ]


class ConceptIndex:
    """Registry concepts addressable by alias key in O(1).

    Only reviewed names (the preferred label and accepted names) are
    indexed: an unreviewed observed name must not become identity evidence.
    Iteration keeps insertion order, so candidate lists stay deterministic.
    """

    def __init__(self, concepts: Iterable[Concept] = ()) -> None:
        self._concepts: Dict[str, Concept] = {}
        self._order: Dict[str, int] = {}
        self._keys: Dict[str, set] = defaultdict(set)
        self._normalized: Dict[str, set] = defaultdict(set)
        self._indexed: Dict[str, Tuple[frozenset, frozenset]] = {}
        for concept in concepts:
            self.add(concept)

    def __iter__(self):
        return iter(list(self._concepts.values()))

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    def get(self, concept_id: str) -> Optional[Concept]:
        return self._concepts.get(concept_id)

    def add(self, concept: Concept) -> None:
        """Insert a concept or replace one with the same identifier."""
        concept_id = concept.concept_id
        self._unindex(concept_id)
        self._concepts[concept_id] = concept
        self._order.setdefault(concept_id, len(self._order))
        aliases = _aliases(concept)
        keys = frozenset().union(
            *(_alias_keys(alias, concept.kind) for alias in aliases)
        )
        normalized = frozenset(normalize_name(alias) for alias in aliases)
        for key in keys:
            self._keys[key].add(concept_id)
        for name in normalized:
            self._normalized[name].add(concept_id)
        self._indexed[concept_id] = (keys, normalized)

    def _unindex(self, concept_id: str) -> None:
        keys, normalized = self._indexed.pop(
            concept_id, (frozenset(), frozenset())
        )
        for key in keys:
            self._keys[key].discard(concept_id)
        for name in normalized:
            self._normalized[name].discard(concept_id)

    def matches(
        self, text: str, groups: list, kind: object = None
    ) -> List[Concept]:
        """Concepts sharing an identity key or an explicit synonym."""
        found = set()
        for key in _alias_keys(text, kind):
            found |= self._keys.get(key, set())
        normalized = normalize_name(text)
        for group in groups:
            names = {normalize_name(name) for name in group["names"]}
            if normalized not in names:
                continue
            for name in names:
                found |= {
                    concept_id
                    for concept_id in self._normalized.get(name, ())
                    if self._concepts[concept_id].kind.value == group["kind"]
                }
        return [
            self._concepts[concept_id]
            for concept_id in sorted(found, key=self._order.__getitem__)
        ]


def _mention_kind(mention: Mention) -> ConceptKind:
    return (
        mention.type_candidates[0]
        if mention.type_candidates
        else ConceptKind.CANDIDATE
    )


def _compatible(mention: Mention, concept: Concept) -> bool:
    return (
        concept.kind in mention.type_candidates
        or ConceptKind.CANDIDATE in mention.type_candidates
    )


def _add_alias(concept: Concept, text: str) -> None:
    normalized = normalize_name(text)
    if normalized in {normalize_name(alias) for alias in _aliases(concept)}:
        return
    concept.names.append(
        ConceptName(
            name_id=stable_id("name", concept.concept_id, normalized),
            text=text,
            normalized_text=normalized,
            name_kind="observed",
            status="provisional",
        )
    )


def resolve_mentions(
    mentions: Sequence[Mention],
    registry: Iterable[Concept],
    semantic: Optional[SemanticDeduplicator] = None,
    semantic_candidates: bool = False,
) -> Tuple[List[Concept], List[ResolutionDecision]]:
    """Resolve mentions against a registry.

    A semantic match is never an identity. By default it makes the mention
    ambiguous; with ``semantic_candidates`` the mention keeps its own
    provisional concept and the match is recorded as a review candidate, so
    claims about it are not lost.

    A ``ConceptIndex`` registry is updated in place: matched concepts gain
    observed names and new provisional concepts are added, so the next
    document sees them. The returned concepts are independent copies that
    later resolutions cannot change while they are being published.
    """
    concepts = (
        registry
        if isinstance(registry, ConceptIndex)
        else ConceptIndex(registry)
    )
    groups = load_catalog("resolver")["explicit_aliases"]
    touched: Dict[str, Concept] = {}
    decisions: List[ResolutionDecision] = []

    for mention in mentions:
        canonical_text = mention.canonical_text or mention.surface_text
        deterministic = [
            concept
            for concept in concepts.matches(
                canonical_text, groups, _mention_kind(mention)
            )
            if _compatible(mention, concept)
        ]
        resolution_id = stable_id(
            "resolution", mention.mention_id, "cascade-v1"
        )

        if len(deterministic) == 1:
            concept = deterministic[0]
            _add_alias(concept, mention.surface_text)
            touched[concept.concept_id] = concept
            decisions.append(
                ResolutionDecision(
                    resolution_id=resolution_id,
                    mention_id=mention.mention_id,
                    status="accepted",
                    concept_id=concept.concept_id,
                    method="normalized_lemma_or_explicit_alias",
                    score=1.0,
                    basis=["deterministic alias match"],
                    review_status="not_required",
                )
            )
            continue

        if len(deterministic) > 1:
            decisions.append(
                ResolutionDecision(
                    resolution_id=resolution_id,
                    mention_id=mention.mention_id,
                    status="ambiguous",
                    candidates=[
                        {"concept_id": item.concept_id, "score": 1.0}
                        for item in deterministic
                    ],
                    method="deterministic_alias_collision",
                    score=1.0,
                    basis=["alias maps to multiple compatible concepts"],
                )
            )
            continue

        semantic_candidate = None
        if semantic is not None:
            match, cosine, model_score = semantic.best_match(
                canonical_text,
                [
                    concept
                    for concept in concepts
                    if _compatible(mention, concept)
                ],
            )
            if match is not None and semantic_candidates:
                semantic_candidate = {
                    "concept_id": match.concept_id,
                    "kind": match.kind.value,
                    "score": model_score,
                    "cosine": cosine,
                }
            elif match is not None:
                touched[match.concept_id] = match
                decisions.append(
                    ResolutionDecision(
                        resolution_id=resolution_id,
                        mention_id=mention.mention_id,
                        status="ambiguous",
                        candidates=[
                            {
                                "concept_id": match.concept_id,
                                "score": model_score,
                            }
                        ],
                        method="embedding_cosine_then_cross_encoder",
                        score=model_score,
                        basis=[
                            f"cosine={cosine:.4f}",
                            f"model={model_score:.4f}",
                        ],
                        review_status="pending",
                    )
                )
                continue

        kind = (
            mention.type_candidates[0]
            if mention.type_candidates
            else ConceptKind.CANDIDATE
        )
        concept_id = stable_id("concept", "provisional", mention.mention_id)
        concept = Concept(
            concept_id=concept_id,
            kind=kind,
            preferred_label=canonical_text,
            status="provisional",
            names=[
                ConceptName(
                    name_id=stable_id(
                        "name",
                        concept_id,
                        normalize_name(mention.surface_text),
                    ),
                    text=mention.surface_text,
                    normalized_text=normalize_name(mention.surface_text),
                    name_kind="observed",
                    status="provisional",
                )
            ],
        )
        concepts.add(concept)
        touched[concept_id] = concept
        decisions.append(
            ResolutionDecision(
                resolution_id=resolution_id,
                mention_id=mention.mention_id,
                status="provisional",
                concept_id=concept_id,
                method=SEMANTIC_CANDIDATE_METHOD
                if semantic_candidate
                else "new_provisional",
                candidates=[semantic_candidate] if semantic_candidate else [],
                score=semantic_candidate["score"]
                if semantic_candidate
                else None,
                basis=[
                    "no deterministic equivalent; new concept requires review",
                    *(
                        [
                            f"semantic candidate cosine="
                            f"{semantic_candidate['cosine']:.4f}"
                        ]
                        if semantic_candidate
                        else []
                    ),
                ],
            )
        )

    return [
        concept.model_copy(deep=True) for concept in touched.values()
    ], decisions


def resolve_exact_mentions(
    mentions: Sequence[Mention], registry: Iterable[Concept]
) -> Tuple[List[Concept], List[ResolutionDecision]]:
    return resolve_mentions(mentions, registry)


class ConceptRegistry:
    """The concept registry shared by all documents of one job.

    Read from the graph once, then kept current in memory: each resolution
    adds its new provisional concepts, so a later (or concurrent) document
    resolves to them instead of re-reading the graph. Resolutions are
    serialized; the CPU-bound matching runs in a worker thread.
    """

    def __init__(self, concepts: Iterable[Concept] = ()) -> None:
        self.index = ConceptIndex(concepts)
        self._lock = None

    def __len__(self) -> int:
        return len(self.index)

    def __iter__(self):
        return iter(self.index)

    async def resolve(
        self,
        mentions: Sequence[Mention],
        semantic: Optional[SemanticDeduplicator] = None,
        semantic_candidates: bool = False,
    ) -> Tuple[List[Concept], List[ResolutionDecision]]:
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await asyncio.to_thread(
                resolve_mentions,
                mentions,
                self.index,
                semantic,
                semantic_candidates,
            )

    def merge(self, concepts: Iterable[Concept]) -> None:
        """Adopt published concept versions (e.g. newly accepted names)."""
        for concept in concepts:
            self.index.add(concept.model_copy(deep=True))
