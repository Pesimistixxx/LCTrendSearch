from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Concept, ConceptKind, ConceptName, Mention, ResolutionDecision, stable_id


def normalize_name(value: str) -> str:
    value = "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char for char in value
    )
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\W_]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def lemmatize_name(value: str) -> str:
    normalized = normalize_name(value)
    try:
        import simplemma
    except ImportError:
        return normalized
    return " ".join(simplemma.lemmatize(token, lang=("en", "ru")) for token in normalized.split())


def alias_keys(value: str) -> set[str]:
    normalized = normalize_name(value)
    lemma = lemmatize_name(value)
    keys = {f"normalized:{normalized}", f"lemma:{lemma}"}
    tokens = lemma.split()
    if len(tokens) > 1:
        keys.add(f"abbreviation:{''.join(token[0] for token in tokens)}")
    elif tokens and 2 <= len(tokens[0]) <= 10:
        keys.add(f"abbreviation:{tokens[0]}")
    return keys


class SemanticDeduplicator:
    """Embedding retrieval followed by a cross-encoder decision."""

    def __init__(
        self,
        cosine_threshold: float = 0.78,
        decision_threshold: float = 0.80,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        decision_model: str = "cross-encoder/stsb-distilroberta-base",
    ) -> None:
        self.cosine_threshold = cosine_threshold
        self.decision_threshold = decision_threshold
        self.embedding_model_name = embedding_model
        self.decision_model_name = decision_model
        self._embedding_tokenizer = None
        self._embedding_model = None
        self._decision_tokenizer = None
        self._decision_model = None
        self._cache: Dict[str, object] = {}

    def _embedding(self, text: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        key = normalize_name(text)
        if key in self._cache:
            return self._cache[key]
        if self._embedding_model is None:
            self._embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self._embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
            self._embedding_model.eval()
        encoded = self._embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            output = self._embedding_model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        vector = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        vector = torch.nn.functional.normalize(vector, dim=1)
        self._cache[key] = vector
        return vector

    def _decision_score(self, left: str, right: str) -> float:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._decision_model is None:
            self._decision_tokenizer = AutoTokenizer.from_pretrained(self.decision_model_name)
            self._decision_model = AutoModelForSequenceClassification.from_pretrained(
                self.decision_model_name
            )
            self._decision_model.eval()
        encoded = self._decision_tokenizer(
            left, right, return_tensors="pt", truncation=True, max_length=128
        )
        with torch.no_grad():
            logits = self._decision_model(**encoded).logits.squeeze()
        if logits.numel() == 1:
            return float(torch.sigmoid(logits).item())
        return float(torch.softmax(logits, dim=-1)[-1].item())

    def best_match(self, text: str, concepts: Sequence[Concept]) -> Tuple[Optional[Concept], float, float]:
        if not concepts:
            return None, 0.0, 0.0
        source = self._embedding(text)
        scored = [
            (float((source @ self._embedding(concept.preferred_label).T).item()), concept)
            for concept in concepts
        ]
        cosine, concept = max(scored, key=lambda item: item[0])
        if cosine < self.cosine_threshold:
            return None, cosine, 0.0
        decision = self._decision_score(text, concept.preferred_label)
        return (concept if decision >= self.decision_threshold else None), cosine, decision


def _aliases(concept: Concept) -> List[str]:
    return [concept.preferred_label, *(name.text for name in concept.names)]


def _compatible(mention: Mention, concept: Concept) -> bool:
    return concept.kind in mention.type_candidates or ConceptKind.CANDIDATE in mention.type_candidates


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
) -> Tuple[List[Concept], List[ResolutionDecision]]:
    concepts = list(registry)
    touched: Dict[str, Concept] = {}
    decisions: List[ResolutionDecision] = []

    for mention in mentions:
        compatible = [concept for concept in concepts if _compatible(mention, concept)]
        mention_keys = alias_keys(mention.surface_text)
        deterministic = [
            concept
            for concept in compatible
            if any(mention_keys & alias_keys(alias) for alias in _aliases(concept))
        ]
        resolution_id = stable_id("resolution", mention.mention_id, "cascade-v1")

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
                    method="normalized_lemma_or_abbreviation",
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
                    candidates=[{"concept_id": item.concept_id, "score": 1.0} for item in deterministic],
                    method="deterministic_alias_collision",
                    score=1.0,
                    basis=["alias maps to multiple compatible concepts"],
                )
            )
            continue

        if semantic is not None:
            match, cosine, model_score = semantic.best_match(mention.surface_text, compatible)
            if match is not None:
                _add_alias(match, mention.surface_text)
                touched[match.concept_id] = match
                decisions.append(
                    ResolutionDecision(
                        resolution_id=resolution_id,
                        mention_id=mention.mention_id,
                        status="accepted",
                        concept_id=match.concept_id,
                        method="embedding_cosine_then_cross_encoder",
                        score=model_score,
                        basis=[f"cosine={cosine:.4f}", f"model={model_score:.4f}"],
                        review_status="not_required",
                    )
                )
                continue

        kind = mention.type_candidates[0] if mention.type_candidates else ConceptKind.CANDIDATE
        concept_id = stable_id("concept", "provisional", mention.mention_id)
        concept = Concept(
            concept_id=concept_id,
            kind=kind,
            preferred_label=mention.surface_text,
            status="provisional",
            names=[
                ConceptName(
                    name_id=stable_id("name", concept_id, normalize_name(mention.surface_text)),
                    text=mention.surface_text,
                    normalized_text=normalize_name(mention.surface_text),
                    name_kind="observed",
                    status="provisional",
                )
            ],
        )
        concepts.append(concept)
        touched[concept_id] = concept
        decisions.append(
            ResolutionDecision(
                resolution_id=resolution_id,
                mention_id=mention.mention_id,
                status="provisional",
                concept_id=concept_id,
                method="new_provisional",
                basis=["no deterministic or model-approved equivalent"],
            )
        )

    return list(touched.values()), decisions


def resolve_exact_mentions(
    mentions: Sequence[Mention], registry: Iterable[Concept]
) -> Tuple[List[Concept], List[ResolutionDecision]]:
    return resolve_mentions(mentions, registry)
