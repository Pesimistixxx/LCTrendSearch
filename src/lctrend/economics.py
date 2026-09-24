from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from .models import (
    Chunk,
    Concept,
    ConceptKind,
    EconomicEvidence,
    Mention,
    ResolutionDecision,
    stable_id,
)
from .resolver import normalize_name


ECONOMIC_PATTERNS = {
    "cost": re.compile(
        r"\b(costs?|prices?|expenses?|affordab\w*|стоимост\w*|цен[аы]?|затрат\w*)\b",
        re.IGNORECASE,
    ),
    "investment": re.compile(
        r"\b(invest\w*|funding|grants?|capital|инвест\w*|финансир\w*|грант\w*)\b",
        re.IGNORECASE,
    ),
    "market": re.compile(
        r"\b(market|revenue|sales|cagr|рын\w*|выручк\w*|продаж\w*)\b",
        re.IGNORECASE,
    ),
    "savings": re.compile(
        r"\b(savings?|payback|profitab\w*|экономи\w*|окупаем\w*|рентабельн\w*)\b",
        re.IGNORECASE,
    ),
    "commercialization": re.compile(
        r"\b(commercial\w*|moneti[sz]\w*|коммерциализац\w*|монетизац\w*)\b",
        re.IGNORECASE,
    ),
}
MONEY_PATTERN = re.compile(
    r"(?:[$€£₽]\s?\d[\d\s.,]*(?:\s?(?:million|billion|млн|млрд))?"
    r"|\d[\d\s.,]*(?:\s?(?:million|billion|млн|млрд))?\s?(?:USD|EUR|RUB|GBP|доллар\w*|евро|рубл\w*))",
    re.IGNORECASE,
)
CURRENCIES = {
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "₽": "RUB",
    "rub": "RUB",
    "руб": "RUB",
    "доллар": "USD",
    "евро": "EUR",
}


def _currency(amount: str) -> Optional[str]:
    lowered = amount.casefold()
    return next((code for token, code in CURRENCIES.items() if token in lowered), None)


def extract_economic_evidence(
    chunks: Iterable[Chunk],
    mentions: Sequence[Mention],
    concepts: Sequence[Concept],
    resolutions: Sequence[ResolutionDecision],
) -> List[EconomicEvidence]:
    """Return only economic sentences that explicitly mention a resolved technology."""
    decisions = {item.mention_id: item.concept_id for item in resolutions if item.concept_id}
    technology_ids = {
        concept.concept_id for concept in concepts if concept.kind == ConceptKind.TECHNOLOGY
    }
    mentions_by_chunk: Dict[str, List[Mention]] = {}
    for mention in mentions:
        if decisions.get(mention.mention_id) in technology_ids:
            mentions_by_chunk.setdefault(mention.chunk_id, []).append(mention)

    evidence: List[EconomicEvidence] = []
    seen = set()
    for chunk in chunks:
        for sentence in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", chunk.text):
            quote = sentence.group().strip()
            if not quote:
                continue
            start = sentence.start() + len(sentence.group()) - len(sentence.group().lstrip())
            end = start + len(quote)
            categories = [name for name, pattern in ECONOMIC_PATTERNS.items() if pattern.search(quote)]
            amount = MONEY_PATTERN.search(quote)
            if not categories and not amount:
                continue
            category = categories[0] if categories else "monetary_value"
            for mention in mentions_by_chunk.get(chunk.chunk_id, []):
                if mention.start < start or mention.end > end:
                    continue
                concept_id = decisions[mention.mention_id]
                key = (concept_id, category, normalize_name(quote))
                if key in seen:
                    continue
                seen.add(key)
                amount_text = amount.group().strip() if amount else None
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
                        currency=_currency(amount_text) if amount_text else None,
                    )
                )
    return evidence
