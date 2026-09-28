from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from time import monotonic, sleep
from typing import (
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from ..core.config import load_catalog
from ..core.models import (
    AMBIGUOUS_COLLISION_METHOD,
    DECLARED_ALIAS_METHOD,
    SEMANTIC_CANDIDATE_METHOD,
    Concept,
    ConceptKind,
    ConceptName,
    Mention,
    ResolutionDecision,
    stable_id,
)
from .lexical import (
    identity_key,
    kind_family,
    lexical_key,
    lexical_tokens,
    settled_kind,
)


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


def context_text(label: str, definition: Optional[str] = None) -> str:
    """The text a concept is embedded and compared as: its name and, when a
    source says what it is, that definition. A bare code ("ML-236B") says
    nothing about its meaning; "ML-236B: inhibitor of cholesterol
    synthesis" does.
    """
    definition = " ".join((definition or "").split())
    return f"{label}: {definition}" if definition else label


def concept_text(concept: Concept) -> str:
    return context_text(concept.preferred_label, concept.definition)


# A quote is evidence, not a name: a surface text longer than the label it
# contains, or longer than any name, is recorded as the label.
_NAME_MAX_TOKENS = 8


def observed_name(surface: str, canonical: Optional[str] = None) -> str:
    """The name a mention adds to its concept."""
    canonical = canonical or surface
    tokens, label = lexical_tokens(surface), lexical_tokens(canonical)
    if len(tokens) > _NAME_MAX_TOKENS or (
        len(tokens) > len(label) and set(label) <= set(tokens)
    ):
        return canonical
    return surface


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
# A transient embedding error (timeout, 429, 5xx) is retried before the
# layer is paused: one slow response must not leave a document unembedded.
EMBEDDING_ATTEMPTS = 3
EMBEDDING_RETRY_SECONDS = 1.0
EMBEDDING_MAX_RETRY_SECONDS = 20.0

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
        # The cross-encoder (local torch) failing is not the embeddings
        # failing: vectors are still stored, only candidates stop.
        self.decision_failure: Optional[str] = None
        self.seeded_model: Optional[str] = None

    def seed(self, labeled: Iterable[Tuple[str, Sequence[float]]]) -> int:
        """Preload label vectors stored in the graph (same model), so a new
        process does not re-embed the whole registry on its first match
        (C-9). Returns how many labels were added."""
        added = 0
        for label, vector in labeled:
            key = normalize_name(label)
            if key and vector and key not in self._cache:
                self._cache[key] = _unit([float(value) for value in vector])
                added += 1
        self.seeded_model = self.embedding_model_name
        return added

    def cached_vector(self, text: str) -> Optional[List[float]]:
        """The unit vector of a text already embedded or seeded, without a
        request; None when it is not in the cache."""
        return self._cache.get(normalize_name(text))

    def available(self) -> bool:
        return (
            self.failure is None
            or monotonic() - self._failed_at >= SEMANTIC_RETRY_SECONDS
        )

    def _transient(self, exc: Exception) -> bool:
        """Every key rate limited (HTTP 429) is a busy moment, not a broken
        layer: this call goes without vectors, the next one tries again.
        Switching the layer off would resolve the next minutes lexically
        and leave provisional duplicates behind."""
        from ..llm.client import rate_limited

        if not rate_limited(exc):
            return False
        logger.info("Embeddings rate limited; this call goes without them")
        return True

    def _fail(self, exc: Exception) -> None:
        self.failure = type(exc).__name__
        self._failed_at = monotonic()
        logger.warning(
            "Semantic layer unavailable (%s); resolving lexically for %.0fs",
            self.failure,
            SEMANTIC_RETRY_SECONDS,
        )

    def embed(
        self, texts: Sequence[str], cache: bool = True
    ) -> Optional[List[List[float]]]:
        """Unit vectors of labels, or None when the layer is unavailable.

        ``cache=False`` for long texts (evidence chunks, search queries):
        they rarely repeat and would only grow the label cache.
        Synchronous: call it from a worker thread, as resolution does.
        """
        if not self.available():
            return None
        try:
            vectors = (
                self._embed(texts) if cache else self._embed_uncached(texts)
            )
        except Exception as exc:
            if self._transient(exc):
                return None
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

            # Reuses GIGACHAT_CREDENTIALS (or the GIGACHAT_KEYS_FILE pool),
            # scope, base URL and CA bundle.
            self._embedder = JsonLLM.from_environment(provider="gigachat")
        from ..core.aio import resolve, run_sync
        from ..llm.client import LLMError

        for attempt in range(EMBEDDING_ATTEMPTS):
            try:
                # Resolution runs in a worker thread, off the event loop.
                return run_sync(
                    resolve(
                        self._embedder.embed(texts, self.embedding_model_name)
                    )
                )
            except LLMError as exc:
                if not exc.retryable or attempt == EMBEDDING_ATTEMPTS - 1:
                    raise
                delay = min(
                    exc.retry_after
                    if exc.retry_after is not None
                    else EMBEDDING_RETRY_SECONDS * 2**attempt,
                    EMBEDDING_MAX_RETRY_SECONDS,
                )
                logger.info(
                    "Embedding request failed (%s); retry %d/%d in %.1fs",
                    exc.code,
                    attempt + 1,
                    EMBEDDING_ATTEMPTS - 1,
                    delay,
                )
                sleep(max(0.0, delay))
        raise AssertionError("unreachable")

    def _compute(self, texts: List[str]) -> List[List[float]]:
        return (
            self._remote_embeddings(texts)
            if self.embedding_provider == "gigachat"
            else self._local_embeddings(texts)
        )

    def _embed_uncached(self, texts: Iterable[str]) -> List[List[float]]:
        texts = list(texts)
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors += [_unit(vector) for vector in self._compute(batch)]
        return vectors

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
        compute = self._compute
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
            if self._transient(exc):
                return None, 0.0, 0.0
            self._fail(exc)
            return None, 0.0, 0.0
        self.failure = None
        return result

    def _best_match(
        self, text: str, concepts: Sequence[Concept]
    ) -> Tuple[Optional[Concept], float, float]:
        source, *targets = self._embed(
            [text, *(concept_text(concept) for concept in concepts)]
        )
        # One matrix product over the registry instead of a Python loop
        # per concept (C-8); vectors are unit length, so dot = cosine.
        scores = np.asarray(targets, dtype=float) @ np.asarray(
            source, dtype=float
        )
        best = int(np.argmax(scores))
        cosine, concept = float(scores[best]), concepts[best]
        if cosine < self.cosine_threshold:
            return None, cosine, 0.0
        if self.decision_failure is not None:
            return None, cosine, 0.0
        try:
            decision = self._decision_score(text, concept_text(concept))
        except Exception as exc:
            self.decision_failure = type(exc).__name__
            logger.warning(
                "Cross-encoder %s unavailable (%s); embeddings are still "
                "stored, semantic review candidates are skipped",
                self.decision_model_name,
                self.decision_failure,
            )
            return None, cosine, 0.0
        return (
            (concept if decision >= self.decision_threshold else None),
            cosine,
            decision,
        )


class AliasGroup(NamedTuple):
    kind: str
    keys: frozenset
    # Identity key of the group's first name: every member shares it.
    canonical: str


@lru_cache(maxsize=1)
def alias_groups() -> Tuple[AliasGroup, ...]:
    """Curated synonym groups compared by identity key, not written form."""
    return tuple(
        AliasGroup(
            group["kind"],
            frozenset().union(
                *(_alias_keys(name, group["kind"]) for name in group["names"])
            ),
            identity_key(group["names"][0], group["kind"]),
        )
        for group in load_catalog("resolver")["explicit_aliases"]
    )


def alias_names(text: str, kind: object) -> List[str]:
    """Curated synonyms of a name (the name itself included)."""
    family = kind_family(kind)
    key = f"key:{identity_key(text, kind)}"
    names = [text]
    for group in load_catalog("resolver")["explicit_aliases"]:
        if kind_family(group["kind"]) == family and key in frozenset().union(
            *(_alias_keys(name, group["kind"]) for name in group["names"])
        ):
            names += group["names"]
    return list(dict.fromkeys(names))


def _group(
    key: str, kind: object, groups: Sequence[AliasGroup]
) -> Optional[AliasGroup]:
    family = kind_family(kind)
    return next(
        (
            group
            for group in groups
            if f"key:{key}" in group.keys and kind_family(group.kind) == family
        ),
        None,
    )


def _preferred(counts: Dict[str, int]) -> str:
    """The most frequent form; ties go to the smallest form, not the first."""
    return min(counts, key=lambda form: (-counts[form], form))


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
            *(_alias_keys(alias, concept.kind) for alias in aliases),
            {f"key:{concept.identity_key}"} if concept.identity_key else (),
        )
        normalized = frozenset(normalize_name(alias) for alias in aliases)
        for key in keys:
            self._keys[key].add(concept_id)
        for name in normalized:
            self._normalized[name].add(concept_id)
        self._indexed[concept_id] = (keys, normalized)

    def named(self, concept: Concept, text: str) -> bool:
        """Whether a reviewed name of the concept is exactly this form."""
        _, normalized = self._indexed.get(concept.concept_id, ((), ()))
        return normalize_name(text) in normalized

    def _unindex(self, concept_id: str) -> None:
        keys, normalized = self._indexed.pop(
            concept_id, (frozenset(), frozenset())
        )
        for key in keys:
            self._keys[key].discard(concept_id)
        for name in normalized:
            self._normalized[name].discard(concept_id)

    def matches(
        self, text: str, groups: Sequence[AliasGroup], kind: object = None
    ) -> List[Concept]:
        """Concepts sharing an identity key or an explicit synonym group."""
        keys = _alias_keys(text, kind)
        found = set()
        for key in keys:
            found |= self._keys.get(key, set())
        for group in groups:
            if not keys & group.keys:
                continue
            for key in group.keys:
                found |= {
                    concept_id
                    for concept_id in self._keys.get(key, ())
                    if kind_family(self._concepts[concept_id].kind)
                    == kind_family(group.kind)
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
    if concept.identity_scope:
        return False
    if kind_family(concept.kind) == "technology":
        return concept.kind in mention.type_candidates
    return kind_family(concept.kind) in {
        kind_family(kind) for kind in mention.type_candidates
    }


def _add_alias(
    concept: Concept,
    text: str,
    name_kind: str = "observed",
    status: str = "provisional",
) -> None:
    normalized = normalize_name(text)
    if not normalized:
        return
    for index, name in enumerate(concept.names):
        if name.normalized_text != normalized:
            continue
        if status == "accepted" and name.status != "accepted":
            # A name the source declared is no longer only observed.
            concept.names[index] = name.model_copy(
                update={"name_kind": name_kind, "status": status}
            )
        return
    if normalized in {normalize_name(alias) for alias in _aliases(concept)}:
        return
    concept.names.append(
        ConceptName(
            name_id=stable_id("name", concept.concept_id, normalized),
            text=text,
            normalized_text=normalized,
            name_kind=name_kind,
            status=status,
        )
    )


def _seed_kind_counts(concept: Concept) -> Dict[str, int]:
    """Kind counts of a concept stored before they were kept: its mentions
    so far are counted under its current kind."""
    return dict(concept.kind_counts) or {
        concept.kind.value: max(1, sum(concept.label_counts.values()))
    }


def concept_identity(
    text: str, kind: ConceptKind, groups: Sequence[AliasGroup] = ()
) -> Tuple[str, ConceptKind]:
    """Identity key and kind of a new concept; a synonym group fixes both."""
    key = identity_key(text, kind)
    group = _group(key, kind, groups or alias_groups())
    if group is None or kind == ConceptKind.CANDIDATE:
        return key, kind
    return group.canonical, kind if kind_family(
        kind
    ) == "technology" else ConceptKind(group.kind)


def _observe(
    concept: Concept, mention: Mention, groups: Sequence[AliasGroup]
) -> None:
    """Record a resolved mention: its name, its form count, its kind and
    what the source says the concept is."""
    _add_alias(
        concept, observed_name(mention.surface_text, mention.canonical_text)
    )
    if not concept.definition and mention.definition:
        concept.definition = mention.definition
    if mention.profile and (
        not concept.profile
        or (
            concept.profile.get("classification_status") != "validated"
            and mention.profile.get("classification_status") == "validated"
        )
    ):
        concept.profile = mention.profile
    if concept.status == "accepted":
        return
    kinds = _seed_kind_counts(concept)
    counts = concept.label_counts or {concept.preferred_label: 1}
    form = mention.canonical_text or mention.surface_text
    counts[form] = counts.get(form, 0) + 1
    concept.label_counts = counts
    concept.preferred_label = _preferred(counts)
    kind = _mention_kind(mention)
    if kind_family(kind) != kind_family(concept.kind):
        return
    kinds[kind.value] = kinds.get(kind.value, 0) + 1
    concept.kind_counts = kinds
    key = concept.identity_key or identity_key(concept.preferred_label)
    # A curated synonym group fixes the kind.
    if (
        kind_family(kind) != "technology"
        and _group(key, concept.kind, groups) is None
    ):
        concept.kind = ConceptKind(settled_kind(kinds, concept.kind))


def _new_concept(
    mention: Mention, text: str, groups: Sequence[AliasGroup]
) -> Concept:
    """A provisional concept identified by kind family and identity key.

    The same name reaches the same concept_id in any document order and in
    concurrent jobs; a curated synonym group fixes the kind and the key.
    """
    key, kind = concept_identity(text, _mention_kind(mention), groups)
    concept_id = stable_id(
        "concept",
        kind.value if kind_family(kind) == "technology" else kind_family(kind),
        key,
    )
    name = observed_name(mention.surface_text, text)
    normalized = normalize_name(name)
    return Concept(
        concept_id=concept_id,
        kind=kind,
        preferred_label=text,
        definition=mention.definition,
        profile=mention.profile,
        status="provisional",
        identity_key=key,
        label_counts={text: 1},
        kind_counts={_mention_kind(mention).value: 1},
        names=[
            ConceptName(
                name_id=stable_id("name", concept_id, normalized),
                text=name,
                normalized_text=normalized,
                name_kind="observed",
                status="provisional",
            )
        ],
    )


def _matches(
    text: str,
    mention: Mention,
    concepts: ConceptIndex,
    groups: Sequence[AliasGroup],
) -> List[Concept]:
    return [
        concept
        for concept in concepts.matches(text, groups, _mention_kind(mention))
        if _compatible(mention, concept)
    ]


def _declared_matches(
    mention: Mention, concepts: ConceptIndex, groups: Sequence[AliasGroup]
) -> List[Concept]:
    """Concepts named by the aliases the source declared for the mention."""
    found: Dict[str, Concept] = {}
    for alias in mention.declared_aliases:
        for concept in _matches(alias, mention, concepts, groups):
            found.setdefault(concept.concept_id, concept)
    return list(found.values())


def _declare(
    concept: Concept,
    mention: Mention,
    concepts: ConceptIndex,
    groups: Sequence[AliasGroup],
) -> List[Dict[str, object]]:
    """Give the concept the names its source equates with it.

    A declared alias becomes an accepted name, so later documents that use
    only that name resolve here. An alias that already names another
    concept is not taken over: the pair becomes a merge candidate.
    """
    candidates: List[Dict[str, object]] = []
    family = kind_family(concept.kind)
    for alias in mention.declared_aliases:
        named = [
            other
            for other in concepts.matches(alias, groups)
            if other.concept_id != concept.concept_id
        ]
        if any(kind_family(other.kind) != family for other in named):
            # "lovastatin (Merck)": a name of a company is not a name of a
            # compound, whatever the parentheses suggest.
            continue
        others = [other for other in named if _compatible(mention, other)]
        if others:
            candidates += [
                {
                    "concept_id": other.concept_id,
                    "kind": other.kind.value,
                    "score": 1.0,
                    "method": DECLARED_ALIAS_METHOD,
                    "alias": alias,
                }
                for other in others
            ]
            continue
        _add_alias(concept, alias, name_kind="declared", status="accepted")
    concepts.add(concept)
    return candidates


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
    groups = alias_groups()
    touched: Dict[str, Concept] = {}
    decisions: List[ResolutionDecision] = []

    if isinstance(semantic, SemanticDeduplicator) and semantic.available():
        embedded = set(load_catalog("resolver")["semantic"]["embedded_kinds"])
        names = [
            context_text(
                mention.canonical_text or mention.surface_text,
                mention.definition,
            )
            for mention in mentions
            if any(kind.value in embedded for kind in mention.type_candidates)
        ]
        if names:
            # One batched request for the document's names instead of one
            # request per mention inside the loop; failures pause the layer.
            semantic.embed(names)

    for original in mentions:
        mention = original
        assessment = mention.entity_assessment
        if assessment is not None and (
            assessment.technology
            or assessment.identity_scope
            or assessment.resolved_kind == ConceptKind.CANDIDATE
        ):
            # Reviewed meanings form their own identity namespace. A legacy
            # alias (including ML) cannot override a contextual definition.
            kind = assessment.resolved_kind
            text = assessment.canonical_name or (
                mention.canonical_text or mention.surface_text
            )
            scope = assessment.identity_scope
            if assessment.decision == "unresolved" or not scope:
                scope = "local:" + assessment.document_version_id
            key = identity_key(text, kind)
            concept_id = stable_id(
                "concept-v3", kind.value, key, identity_key(scope)
            )
            concept = concepts.get(concept_id)
            if concept is None and assessment.decision != "unresolved":
                # Explicitly reviewed merges may keep a different stable ID.
                # Only accepted names within the same meaning can redirect it.
                equivalent = [
                    item
                    for item in {
                        item.concept_id: item
                        for name in [text, *mention.declared_aliases]
                        for item in concepts.matches(name, [], kind)
                    }.values()
                    if item.kind == kind
                    and item.identity_scope
                    and identity_key(item.identity_scope)
                    == identity_key(scope)
                    and (kind != ConceptKind.TECHNOLOGY or item.technology)
                ]
                if len(equivalent) == 1:
                    concept = equivalent[0]
                    concept_id = concept.concept_id
            if concept is None:
                concept = Concept(
                    concept_id=concept_id,
                    kind=kind,
                    preferred_label=text,
                    identity_key=key,
                    identity_scope=scope,
                    status="accepted"
                    if assessment.decision != "unresolved"
                    else "provisional",
                    technology=assessment.technology,
                    profile=mention.profile,
                    definition=assessment.technology.definition
                    if assessment.technology
                    else None,
                )
            _add_alias(concept, mention.surface_text)
            if assessment.decision != "unresolved":
                for alias in mention.declared_aliases:
                    _add_alias(
                        concept, alias, name_kind="declared",
                        status="accepted",
                    )
            concepts.add(concept)
            touched[concept_id] = concept
            decisions.append(
                ResolutionDecision(
                    resolution_id=stable_id(
                        "resolution", mention.mention_id, "meaning-v3"
                    ),
                    mention_id=mention.mention_id,
                    concept_id=concept_id,
                    status="accepted"
                    if assessment.decision != "unresolved"
                    else "provisional",
                    method="reviewed_entity_meaning",
                    basis=[assessment.assessment_id, assessment.reason],
                    review_status="reviewed",
                )
            )
            continue
        canonical_text = mention.canonical_text or mention.surface_text
        deterministic = _matches(canonical_text, mention, concepts, groups)
        method = "normalized_lemma_or_explicit_alias"
        if not deterministic:
            # "compactin (ML-236B)": a name the source equates with this one
            # identifies the concept when the label alone does not.
            declared = _declared_matches(mention, concepts, groups)
            if len(declared) == 1:
                deterministic, method = declared, DECLARED_ALIAS_METHOD
        resolution_id = stable_id(
            "resolution", mention.mention_id, "cascade-v1"
        )

        if len(deterministic) > 1:
            # Duplicates share a key; the one reviewed under exactly this
            # written form is the identity.
            exact = [
                concept
                for concept in deterministic
                if concepts.named(concept, canonical_text)
            ]
            if len(exact) == 1:
                deterministic = exact

        if len(deterministic) == 1:
            concept = deterministic[0]
            _observe(concept, mention, groups)
            if method == DECLARED_ALIAS_METHOD:
                # The source equated its label with a name of the concept:
                # the label names it from now on too.
                _add_alias(
                    concept,
                    canonical_text,
                    name_kind="declared",
                    status="accepted",
                )
            declared_candidates = _declare(concept, mention, concepts, groups)
            touched[concept.concept_id] = concept
            decisions.append(
                ResolutionDecision(
                    resolution_id=resolution_id,
                    mention_id=mention.mention_id,
                    status="accepted",
                    concept_id=concept.concept_id,
                    method=method,
                    candidates=declared_candidates,
                    score=1.0,
                    basis=[
                        "alias declared by the source"
                        if method == DECLARED_ALIAS_METHOD
                        else "deterministic alias match"
                    ],
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
                    # Every candidate keeps the mention (an ambiguous
                    # MENTIONS link) until the duplicates are merged.
                    candidates=[
                        {
                            "concept_id": item.concept_id,
                            "kind": item.kind.value,
                            "score": 1.0,
                        }
                        for item in deterministic
                    ],
                    method=AMBIGUOUS_COLLISION_METHOD,
                    score=1.0,
                    basis=["alias maps to multiple compatible concepts"],
                )
            )
            continue

        semantic_candidate = None
        if semantic is not None:
            match, cosine, model_score = semantic.best_match(
                context_text(canonical_text, mention.definition),
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

        concept = _new_concept(mention, canonical_text, groups)
        concept_id = concept.concept_id
        existing = concepts.get(concept_id)
        if existing is not None:
            # Same family and key: the same identity, even when no reviewed
            # name of the existing concept matched.
            _observe(existing, mention, groups)
            concept = existing
        concepts.add(concept)
        declared_candidates = _declare(concept, mention, concepts, groups)
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
                candidates=[
                    *([semantic_candidate] if semantic_candidate else []),
                    *declared_candidates,
                ],
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
