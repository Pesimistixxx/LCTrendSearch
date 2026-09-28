"""Literal evidence and document-local reference gates, separate from LLM
review.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ..core.config import load_catalog
from ..core.models import Chunk, DocumentEnvelope
from ..extraction.lexical import lexical_tokens
from ..extraction.resolver import alias_names
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


def _occurrences(text: str, quote: str) -> List[int]:
    # Include overlapping occurrences. 'aa' in 'aaa' has two possible anchors.
    starts = []
    cursor = 0
    while (cursor := text.find(quote, cursor)) != -1:
        starts.append(cursor)
        cursor += 1
    return starts


def _anchor(
    span: SourceSpan,
    chunks: Dict[str, Chunk],
    visible: Set[str],
    first_occurrence: bool = False,
) -> Tuple[SourceSpan, Optional[Dict[str, Any]]]:
    """Anchor a literal quote; return the span and an optional audit note.

    Wrong model offsets fall back to literal search (nearest occurrence).
    A repeated quote without offsets is ambiguous, unless
    ``first_occurrence`` allows the first one: an entity name repeated in a
    chunk names the same entity at each occurrence.
    """
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
    note = None
    if span.start is not None or span.end is not None:
        if (
            span.start is None
            or span.end is None
            or span.start < 0
            or span.end <= span.start
            or span.end > len(text)
        ):
            raise ValueError("invalid_offsets")
        if text[span.start : span.end] == span.quote:
            start = span.start
        else:
            starts = _occurrences(text, span.quote)
            if not starts:
                raise ValueError("quote_offset_mismatch")
            start = min(starts, key=lambda item: abs(item - span.start))
            note = ("quote_offsets_corrected", len(starts))
    else:
        starts = _occurrences(text, span.quote)
        if not starts:
            raise ValueError("quote_not_found")
        if len(starts) > 1:
            if not first_occurrence:
                raise ValueError("ambiguous_quote")
            note = ("ambiguous_quote_first_occurrence", len(starts))
        start = starts[0]
    anchored = span.model_copy(
        update={"start": start, "end": start + len(span.quote)}
    )
    _require_source_mapping(anchored, chunk)
    if note is None:
        return anchored, None
    code, occurrences = note
    return anchored, {
        "chunk_id": span.chunk_id,
        "code": code,
        "occurrences": occurrences,
        "start": start,
    }


def _names_in(names: Iterable[str], texts: Iterable[str]) -> bool:
    """Whether a name occurs in a text as a run of identity-key tokens.

    Comparing keys, not strings, lets case, plural and Russian inflection
    differ ("Германия" names "Германии").
    """
    phrases = [lexical_tokens(name) for name in names]
    phrases = [phrase for phrase in phrases if phrase]
    for text in texts:
        tokens = lexical_tokens(text)
        for phrase in phrases:
            size = len(phrase)
            if any(
                tokens[start : start + size] == phrase
                for start in range(len(tokens) - size + 1)
            ):
                return True
    return False


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


def _currency_grounded(currency: str, raw: str, aliases: dict) -> bool:
    """A symbol can remain ambiguous; a bare dollar never proves USD.

    Whole-word boundaries prevent a code such as TRY from matching an
    ordinary word. Symbols keep their original meaning without FX inference.
    """
    for alias in aliases.get(currency, []):
        left = r"(?<!\w)" if alias[0].isalnum() else ""
        right = r"(?!\w)" if alias[-1].isalnum() else ""
        if re.search(left + re.escape(alias) + right, raw, re.IGNORECASE):
            return True
    return False


def validate_local_extraction(
    document: DocumentEnvelope,
    extraction: Extraction,
    visible_ids: Iterable[str],
    notes: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Extraction, Dict[str, List[str]]]:
    """Return anchored candidates plus issue gates, never acceptance decisions.

    Consumers must exclude every entity/claim with issues from
    review/compilation.
    Invalid records remain in the copy for an audit trail. Validation does not
    mutate the caller's extraction and never silently edits a quote or number.
    Non-blocking anchoring choices (first of repeated entity names, offsets
    corrected by literal search) are appended to ``notes`` for the audit.
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
    grounded_kinds = set(schema["grounded_label_kinds"])
    countries = load_catalog("countries")
    country_codes = set(countries["iso_alpha2"])
    entity_counts = Counter(entity.local_id for entity in result.entities)
    claim_counts = Counter(claim.claim_id for claim in result.claims)
    entities = {entity.local_id: entity for entity in result.entities}
    issues: Dict[str, List[str]] = {}

    def add(key: str, issue: str) -> None:
        issues.setdefault(key, []).append(issue)

    def anchor_spans(
        key: str, spans: List[SourceSpan], first_occurrence: bool = False
    ) -> List[SourceSpan]:
        anchored = []
        for span in spans:
            try:
                value, note = _anchor(
                    span, chunks, visible, first_occurrence
                )
            except ValueError as exc:
                add(key, str(exc))
                anchored.append(span.model_copy(deep=True))
                continue
            anchored.append(value)
            if note is not None and notes is not None:
                notes.append({"item": key, **note})
        return anchored

    for entity in result.entities:
        key = "entity:" + entity.local_id
        if entity_counts[entity.local_id] > 1:
            add(key, "duplicate_entity_id")
        if not entity.local_id.strip() or not entity.label.strip():
            add(key, "empty_entity_identifier_or_label")
        if entity.kind.value == "Country" and not entity.country_code:
            add(key, "country_code_missing")
        if entity.country_code is not None:
            if entity.kind.value != "Country":
                add(key, "country_code_on_non_country")
            elif (
                not re.fullmatch(
                    schema["country_code_pattern"], entity.country_code
                )
                or entity.country_code not in country_codes
            ):
                add(key, "invalid_country_code")
            elif not _names_in(
                countries["names"].get(entity.country_code, []),
                [entity.label, *(span.quote for span in entity.evidence)],
            ):
                # The quoted country must be the coded one: a quote of
                # "Германии" cannot become US.
                add(key, "country_code_mismatch")
        entity.evidence = anchor_spans(
            key, entity.evidence, first_occurrence=True
        )
        # The label names what the source names: its quote or chunk must
        # contain it (or a curated synonym), or it is a phantom entity.
        grounding = [
            text
            for span in entity.evidence
            for text in (
                span.quote,
                chunks[span.chunk_id].text
                if span.chunk_id in chunks
                else "",
            )
        ]
        if (
            entity.kind.value in grounded_kinds
            and entity.label.strip()
            and not _names_in(
                alias_names(entity.label, entity.kind.value), grounding
            )
        ):
            add(key, "label_not_grounded")

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
        if rule:
            for name in rule.get("required_qualifiers", []):
                if claim.qualifiers.get(name) in (None, ""):
                    add(key, "missing_required_qualifier:" + name)
            for name, allowed in rule.get("qualifier_enums", {}).items():
                if (
                    claim.qualifiers.get(name) is not None
                    and claim.qualifiers[name] not in allowed
                ):
                    add(key, "invalid_qualifier:" + name)
            # A maturity number must follow its marker. Six devices or six
            # months of tests cannot silently become TRL 6.
            for name, (low, high) in rule.get(
                "grounded_integer_qualifiers", {}
            ).items():
                number = claim.qualifiers.get(name)
                if number is None:
                    continue
                if (
                    isinstance(number, bool)
                    or not isinstance(number, int)
                    or not low <= number <= high
                ):
                    add(key, "invalid_qualifier:" + name)
                else:
                    marker = rule.get("integer_qualifier_markers", {}).get(
                        name, r"(?<!\d)"
                    )
                    if not any(
                        re.search(
                            marker + rf"{number}(?![\d.,]\d|\d)",
                            quote,
                            re.IGNORECASE,
                        )
                        for quote in quotes
                    ):
                        add(key, "qualifier_not_grounded:" + name)
        for value in claim.values:
            contract = (rule or {}).get("value_contract", {})
            for field in contract.get("required_fields", []):
                if value.get(field) is None or value.get(field) == "":
                    add(key, "missing_value_field:" + field)
            raw = value.get("raw")
            grounded = (
                isinstance(raw, str)
                and bool(raw.strip())
                and any(raw in quote for quote in quotes)
            )
            if not grounded:
                add(key, "value_not_grounded")
            for field in contract.get("grounded_fields", []):
                item = value.get(field)
                if item is not None and (
                    not isinstance(item, str)
                    or not item.strip()
                    or not any(item in quote for quote in quotes)
                ):
                    add(key, "value_field_not_grounded:" + field)
            if "currency" in contract.get("required_fields", []):
                currency = value.get("currency")
                aliases = schema.get("currency_aliases", {})
                if not isinstance(currency, str) or currency not in aliases:
                    add(key, "invalid_currency")
                elif not isinstance(raw, str) or not _currency_grounded(
                    currency, raw, aliases
                ):
                    add(key, "currency_not_grounded")
            numeric = value.get("value")
            if contract.get("numeric_value"):
                try:
                    if isinstance(numeric, bool) or not isinstance(
                        numeric, (int, float, Decimal, str)
                    ):
                        raise InvalidOperation
                    Decimal(str(numeric))
                except InvalidOperation:
                    add(key, "invalid_numeric_value")
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
