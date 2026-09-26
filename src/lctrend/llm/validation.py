"""Literal evidence and document-local reference gates, separate from LLM
review.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Set, Tuple

from ..core.config import load_catalog
from ..core.models import Chunk, DocumentEnvelope
from .contracts import Extraction, Review, SourceSpan


def _require_source_mapping(span: SourceSpan, chunk: Chunk) -> None:
    """Exclude generated text when the adapter supplies explicit source slices.

    Offsets refer to the adapter's decoded source stream, not artifact bytes.
    Native chunks without this optional mapping retain literal anchoring.
    """
    if "source_segments" not in chunk.locator:
        return
    raw_segments = chunk.locator["source_segments"]
    if not isinstance(raw_segments, list):
        raise ValueError("invalid_source_segments")
    segments = []
    fields = ("chunk_start", "chunk_end", "source_start", "source_end")
    for segment in raw_segments:
        if not isinstance(segment, dict) or any(
            type(segment.get(field)) is not int for field in fields
        ):
            raise ValueError("invalid_source_segments")
        start, end, source_start, source_end = (
            segment[field] for field in fields
        )
        if (
            not 0 <= start < end <= len(chunk.text)
            or not 0 <= source_start < source_end
            or end - start != source_end - source_start
        ):
            raise ValueError("invalid_source_segments")
        segments.append((start, end, source_start, source_end))
    segments.sort()
    if any(left[1] > right[0] for left, right in zip(segments, segments[1:])):
        raise ValueError("invalid_source_segments")

    cursor = span.start
    previous_source_end = None
    for start, end, source_start, _ in segments:
        if end <= span.start or start >= span.end:
            continue
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start != cursor:
            raise ValueError("quote_not_mapped_to_source")
        mapped_start = source_start + overlap_start - start
        if (
            previous_source_end is not None
            and mapped_start != previous_source_end
        ):
            raise ValueError("noncontiguous_source_quote")
        previous_source_end = mapped_start + overlap_end - overlap_start
        cursor = overlap_end
    if cursor != span.end:
        raise ValueError("quote_not_mapped_to_source")


def _anchor(
    span: SourceSpan, chunks: Dict[str, Chunk], visible: Set[str]
) -> SourceSpan:
    if span.chunk_id not in chunks:
        raise ValueError("unknown_evidence_chunk")
    if span.chunk_id not in visible:
        raise ValueError("evidence_chunk_not_visible")
    chunk = chunks[span.chunk_id]
    if chunk.parse_status == "rejected":
        raise ValueError("evidence_chunk_parse_rejected")
    text = chunk.text
    if not span.quote.strip():
        raise ValueError("empty_quote")
    if span.start is not None or span.end is not None:
        if (
            span.start is None
            or span.end is None
            or span.start < 0
            or span.end <= span.start
            or span.end > len(text)
        ):
            raise ValueError("invalid_offsets")
        if text[span.start : span.end] != span.quote:
            raise ValueError("quote_offset_mismatch")
        anchored = span.model_copy(deep=True)
    else:
        # Include overlapping occurrences. 'aa' in 'aaa' has two possible
        # anchors.
        starts = []
        cursor = 0
        while (cursor := text.find(span.quote, cursor)) != -1:
            starts.append(cursor)
            cursor += 1
        if len(starts) != 1:
            raise ValueError(
                "quote_not_found" if not starts else "ambiguous_quote"
            )
        anchored = span.model_copy(
            update={"start": starts[0], "end": starts[0] + len(span.quote)}
        )
    _require_source_mapping(anchored, chunk)
    return anchored


def _entity_refs(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "entity_ref":
                yield item
            else:
                yield from _entity_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _entity_refs(item)


def validate_local_extraction(
    document: DocumentEnvelope,
    extraction: Extraction,
    visible_ids: Iterable[str],
) -> Tuple[Extraction, Dict[str, List[str]]]:
    """Return anchored candidates plus issue gates, never acceptance decisions.

    Consumers must exclude every entity/claim with issues from
    review/compilation.
    Invalid records remain in the copy for an audit trail. Validation does not
    mutate the caller's extraction and never silently edits a quote or number.
    """
    # Python mode preserves nonfinite numbers for rejection. JSON serialization
    # may turn NaN/Infinity into null, hiding an invalid model-supplied value.
    result = Extraction.model_validate(
        extraction.model_dump(mode="python")
    ).model_copy(deep=True)
    chunks = {chunk.chunk_id: chunk for chunk in document.chunks}
    visible = set(visible_ids)
    if not visible.issubset(chunks):
        raise ValueError("visible_ids references unknown chunks")
    schema = load_catalog("llm_schema")
    entity_counts = Counter(entity.local_id for entity in result.entities)
    claim_counts = Counter(claim.claim_id for claim in result.claims)
    entities = {entity.local_id: entity for entity in result.entities}
    issues: Dict[str, List[str]] = {}

    def add(key: str, issue: str) -> None:
        issues.setdefault(key, []).append(issue)

    def anchor_spans(key: str, spans: List[SourceSpan]) -> List[SourceSpan]:
        anchored = []
        for span in spans:
            try:
                anchored.append(_anchor(span, chunks, visible))
            except ValueError as exc:
                add(key, str(exc))
                anchored.append(span.model_copy(deep=True))
        return anchored

    for entity in result.entities:
        key = "entity:" + entity.local_id
        if entity_counts[entity.local_id] > 1:
            add(key, "duplicate_entity_id")
        if not entity.local_id.strip() or not entity.label.strip():
            add(key, "empty_entity_identifier_or_label")
        entity.evidence = anchor_spans(key, entity.evidence)

    for claim in result.claims:
        key = "claim:" + claim.claim_id
        if claim_counts[claim.claim_id] > 1:
            add(key, "duplicate_claim_id")
        if not claim.claim_id.strip():
            add(key, "empty_claim_id")
        rule = schema["predicates"].get(claim.predicate)
        if rule is None:
            add(key, "unsupported_predicate")
        else:
            if not set(rule["required_roles"]).issubset(claim.roles):
                add(key, "missing_required_roles")
            if not set(claim.roles).issubset(rule["allowed_roles"]):
                add(key, "unsupported_role")
        for role, entity_id in claim.roles.items():
            entity = entities.get(entity_id)
            if entity is None or "entity:" + entity_id in issues:
                add(key, "invalid_entity:" + entity_id)
            elif (
                rule
                and role in rule["role_types"]
                and entity.kind.value not in rule["role_types"][role]
            ):
                add(key, "role_type_mismatch:" + role)
        for entity_id in _entity_refs([claim.qualifiers, claim.values]):
            if not isinstance(entity_id, str):
                add(key, "invalid_entity_ref")
            elif entity_id not in entities or "entity:" + entity_id in issues:
                add(key, "invalid_entity_ref:" + entity_id)
        claim.evidence = anchor_spans(key, claim.evidence)
        if (
            claim.predicate in schema["measurement_predicates"]
            and not claim.values
        ):
            add(key, "measurement_values_missing")
        quotes = [span.quote for span in claim.evidence]
        for value in claim.values:
            raw = value.get("raw")
            grounded = (
                isinstance(raw, str)
                and bool(raw.strip())
                and any(raw in quote for quote in quotes)
            )
            if not grounded:
                add(key, "value_not_grounded")
            numeric = value.get("value")
            if isinstance(numeric, bool):
                add(key, "invalid_numeric_value")
            elif isinstance(numeric, (int, float, Decimal, str)):
                try:
                    number = Decimal(str(numeric))
                    if not number.is_finite():
                        add(key, "nonfinite_numeric_value")
                    elif grounded:
                        reported = [
                            Decimal(token.replace(",", "."))
                            for token in re.findall(
                                schema["number_pattern"], raw
                            )
                        ]
                        if number not in reported:
                            add(key, "number_not_in_raw_value")
                except InvalidOperation:
                    # Categorical string measurements remain literal strings.
                    # Numeric-looking strings, including NaN/Infinity, are
                    # checked above.
                    if not isinstance(numeric, str):
                        add(key, "invalid_numeric_value")
    return result, {key: sorted(set(value)) for key, value in issues.items()}


def validate_review(review: Review, expected_ids: Iterable[str]) -> Review:
    """Require exactly one decision for each eligible current claim."""
    result = Review.model_validate(
        review.model_dump(mode="python")
    ).model_copy(deep=True)
    expected = list(expected_ids)
    actual = [item.claim_id for item in result.items]
    if len(expected) != len(set(expected)):
        raise ValueError("Expected review IDs must be unique")
    if len(actual) != len(set(actual)):
        raise ValueError("Reviewer returned duplicate claim IDs")
    if set(actual) != set(expected):
        raise ValueError(
            "Reviewer must return exactly one item for every expected claim ID"
        )
    if any(not item.reason.strip() for item in result.items):
        raise ValueError("Reviewer must explain every decision")
    return result
