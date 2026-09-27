from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

from ..core.config import load_catalog
from ..core.models import (
    Chunk,
    Concept,
    ConceptKind,
    EconomicEvidence,
    Mention,
    ResolutionDecision,
    stable_id,
)
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
    digits = re.sub(r"\s", "", match.group()).rstrip(".,")
    if "," in digits and "." in digits:
        digits = digits.replace(",", "")
    elif "," in digits:
        digits = (
            digits.replace(",", "")
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+", digits)
            else digits.replace(",", ".")
        )
    if digits.count(".") > 1:
        if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", digits):
            return None
        digits = digits.replace(".", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    lowered = amount.casefold()
    for word, factor in scales.items():
        if re.search(r"\b" + re.escape(word), lowered):
            return value * factor
    return value


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
