from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence

from ..core.config import load_catalog
from ..core.models import (
    Assertion,
    Chunk,
    Concept,
    ConceptKind,
    DocumentEnvelope,
    EconomicEvidence,
    Mention,
    ResolutionDecision,
    stable_id,
)
from ..core.numbers import parse_number
from .assertions import claim_context, resolved_ids
from .resolver import normalize_name


def _currency(amount: str, currencies: dict[str, str]) -> Optional[str]:
    lowered = amount.casefold()
    return next(
        (code for token, code in currencies.items() if token in lowered), None
    )


def amount_value(amount: str, scales: dict[str, float]) -> Optional[float]:
    """Read "$1,5 млн" or "2.3 billion EUR" as a number in its currency.

    Returns None when the digits are ambiguous rather than guessing.
    """
    match = re.search(r"\d[\d\s.,]*", amount)
    if not match:
        return None
    number = parse_number(match.group())
    if number is None:
        return None
    value = float(number)
    lowered = amount.casefold()
    for word, factor in scales.items():
        if re.search(r"\b" + re.escape(word), lowered):
            return value * factor
    return value


def _reviewed_amount(value: float, raw: str, scales: dict) -> Optional[float]:
    """Use the reviewed numeric reading, preserving its quoted scale."""
    for word, factor in scales.items():
        if re.search(r"\b" + re.escape(word), raw.casefold()):
            value *= factor
            break
    return value if math.isfinite(value) else None


def economic_evidence_from_assertions(
    document: DocumentEnvelope,
    assertions: Sequence[Assertion],
    concepts: Sequence[Concept],
) -> List[EconomicEvidence]:
    """Project reviewed financial facts; regex candidates remain separate.

    Values retain the source currency, period and unit. A quoted scale may
    normalize its numeric magnitude, but there is no currency conversion or
    extrapolation of a forecast into an observed outcome.
    """
    predicates = load_catalog("llm_schema").get("economic_predicates", {})
    rules = load_catalog("extraction")["economics"]
    technologies = {
        concept.concept_id
        for concept in concepts
        if concept.kind == ConceptKind.TECHNOLOGY
    }
    chunks = {chunk.chunk_id: chunk for chunk in document.chunks}
    output = []
    seen = set()
    for assertion in assertions:
        category = predicates.get(assertion.predicate)
        subject = assertion.roles.get("subject")
        if (
            assertion.predicate not in predicates
            or not category
            or subject not in technologies
            or assertion.status != "accepted"
            or assertion.verification_status != "supported"
            or assertion.polarity != "affirmed"
            or assertion.modality not in {"reported", "observed"}
        ):
            continue
        for value in assertion.values:
            raw = value.get("raw")
            number = value.get("value")
            if (
                not isinstance(raw, str)
                or not raw.strip()
                or isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number < 0
                or not isinstance(value.get("currency"), str)
                or not value["currency"].strip()
            ):
                continue
            for span in assertion.evidence:
                chunk = chunks.get(span.chunk_id)
                if (
                    chunk is None
                    or not (0 <= span.start < span.end <= len(chunk.text))
                    or chunk.text[span.start : span.end] != span.quote
                    or raw not in span.quote
                    or any(
                        value.get(field) is not None
                        and (
                            not isinstance(value[field], str)
                            or value[field] not in span.quote
                        )
                        for field in ("unit", "period")
                    )
                ):
                    continue
                key = (
                    assertion.assertion_id,
                    span.chunk_id,
                    span.start,
                    span.end,
                    raw,
                )
                if key in seen:
                    continue
                seen.add(key)
                output.append(
                    EconomicEvidence(
                        evidence_id=stable_id(
                            "reviewed-economic-evidence", *key
                        ),
                        technology_concept_id=subject,
                        assertion_id=assertion.assertion_id,
                        chunk_id=span.chunk_id,
                        category=category,
                        quote=span.quote,
                        start=span.start,
                        end=span.end,
                        amount_text=raw,
                        amount_value=_reviewed_amount(
                            float(number), raw, rules["scales"]
                        ),
                        currency=value.get("currency"),
                        unit=value.get("unit"),
                        period=value.get("period"),
                        polarity=assertion.polarity,
                        modality=assertion.modality,
                        confidence=assertion.extraction_confidence,
                        status="accepted",
                    )
                )
    return output


def extract_economic_evidence(
    chunks: Iterable[Chunk],
    mentions: Sequence[Mention],
    concepts: Sequence[Concept],
    resolutions: Sequence[ResolutionDecision],
) -> List[EconomicEvidence]:
    """Return candidates only when an economic clause names one technology.

    Several possible owners are skipped; one amount is never distributed among
    all technologies in a sentence. Computational resources are not money.
    """
    settings = load_catalog("extraction")
    rules = settings["economics"]
    categories = {
        name: re.compile(pattern, re.IGNORECASE)
        for name, pattern in rules["categories"].items()
    }
    money = re.compile(rules["money_pattern"], re.IGNORECASE)
    non_monetary = re.compile(rules["non_monetary_cost"], re.IGNORECASE)
    decisions = resolved_ids(resolutions)
    technology_ids = {
        concept.concept_id
        for concept in concepts
        if concept.kind == ConceptKind.TECHNOLOGY
    }
    by_chunk: dict[str, list[Mention]] = {}
    for mention in mentions:
        if decisions.get(mention.mention_id) in technology_ids:
            by_chunk.setdefault(mention.chunk_id, []).append(mention)

    evidence: list[EconomicEvidence] = []
    seen = set()
    for chunk in chunks:
        for sentence in re.finditer(settings["sentence_pattern"], chunk.text):
            boundaries = [
                (match.start(), match.end())
                for match in re.finditer(
                    rules["clause_split_pattern"],
                    sentence.group(),
                    re.IGNORECASE,
                )
            ]
            clause_start = 0
            for stop, next_start in [
                *boundaries,
                (len(sentence.group()), len(sentence.group())),
            ]:
                raw = sentence.group()[clause_start:stop]
                quote = raw.strip()
                start = (
                    sentence.start()
                    + clause_start
                    + len(raw)
                    - len(raw.lstrip())
                )
                end = start + len(quote)
                clause_start = next_start
                if not quote:
                    continue
                owners = {
                    decisions[item.mention_id]
                    for item in by_chunk.get(chunk.chunk_id, [])
                    if start <= item.start < item.end <= end
                }
                if len(owners) != 1:
                    continue
                local_mentions = [
                    item
                    for item in by_chunk.get(chunk.chunk_id, [])
                    if start <= item.start < item.end <= end
                ]
                directly_related = any(
                    (
                        not chunk.text[start : item.start].strip()
                        and re.search(
                            rules["ownership_suffix"],
                            chunk.text[item.end : end],
                            re.IGNORECASE,
                        )
                    )
                    or re.fullmatch(
                        rules["ownership_prefix"],
                        chunk.text[start : item.start].lstrip(),
                        re.IGNORECASE,
                    )
                    for item in local_mentions
                )
                if not directly_related:
                    continue
                category_matches = [
                    name
                    for name, pattern in categories.items()
                    if pattern.search(quote)
                ]
                amounts = list(money.finditer(quote))
                if len(amounts) > 1:
                    continue
                if non_monetary.search(quote) and not amounts:
                    category_matches = [
                        name for name in category_matches if name != "cost"
                    ]
                if not category_matches and not amounts:
                    continue
                category = (
                    category_matches[0]
                    if category_matches
                    else "monetary_value"
                )
                concept_id = next(iter(owners))
                key = (
                    concept_id,
                    category,
                    chunk.chunk_id,
                    start,
                    end,
                    normalize_name(quote),
                )
                if key in seen:
                    continue
                seen.add(key)
                amount_text = (
                    amounts[0].group().strip().rstrip(".,")
                    if amounts
                    else None
                )
                polarity, modality = claim_context(quote, settings)
                evidence.append(
                    EconomicEvidence(
                        evidence_id=stable_id("economic-evidence", *key),
                        technology_concept_id=concept_id,
                        chunk_id=chunk.chunk_id,
                        category=category,
                        quote=quote,
                        start=start,
                        end=end,
                        amount_text=amount_text,
                        amount_value=amount_value(amount_text, rules["scales"])
                        if amount_text
                        else None,
                        currency=_currency(amount_text, rules["currencies"])
                        if amount_text
                        else None,
                        polarity=polarity,
                        modality=modality,
                        confidence=None,
                        status="candidate",
                    )
                )
    return evidence
