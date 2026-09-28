"""``GET /api/search`` response from the graph, in the frontend contract.

The query selects a scope: domains whose names or aliases
(``sources.json → domains``) match its words, with their child domains;
otherwise technologies whose label words match; otherwise every candidate,
with a note. Scores and z-scores always come from all candidates at T, so a
technology keeps its score whatever the query.

STATUS: draft. The answer logic is not done yet (see ``lctrend.ranking``);
this is prepared functionality, not a finished result.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from datetime import date
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
)

from ..core.config import load_catalog
from ..core.models import stable_id
from ..graph.features import months_before
from ..graph.temporal import CODE, PACKAGE, PATENT, SCHOLARLY, TemporalCorpus
from .scoring import Ranking, evidence_quotes, independent_sources
from .scoring import rank_snapshot as _rank_snapshot

STAGES = {
    1: "концепция",
    2: "лабораторная проверка",
    3: "прототипы",
    4: "пилоты",
}
DOCUMENT_TYPES = {
    SCHOLARLY: "Научная публикация",
    CODE: "Репозиторий кода",
    PACKAGE: "Пакет",
    PATENT: "Патент",
}


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", text.casefold().replace("ё", "е"))


def _word_match(query: str, name: str) -> bool:
    """Same word up to a short inflected ending (финтехе = финтех)."""
    if min(len(query), len(name)) < 4:
        return query == name
    common = 0
    for left, right in zip(query, name):
        if left != right:
            break
        common += 1
    return common >= max(4, min(len(query), len(name)) - 2)


def _phrase_match(query: List[str], phrase: str) -> bool:
    words = _words(phrase)
    return bool(words) and all(
        any(_word_match(item, word) for item in query) for word in words
    )


def match_domains(query: str) -> Dict[str, str]:
    """Domain name -> domain_id for every domain the query names, plus
    their child domains."""
    words = _words(query)
    domains = load_catalog("sources")["domains"]
    names = {
        domain["name"]
        for domain in domains
        if any(
            _phrase_match(words, phrase)
            for phrase in (domain["name"], *domain.get("aliases", []))
        )
    }
    grown = True
    while grown:
        children = {
            domain["name"]
            for domain in domains
            if domain.get("parent_name") in names
        }
        grown = not children <= names
        names |= children
    return {name: stable_id("domain", name) for name in sorted(names)}


def _scope(view, query: str):
    """(scope, matched names, technology ids) for a query at T."""
    domains = match_domains(query)
    if domains:
        wanted = set(domains.values())
        ids = {
            technology_id
            for technology_id, technology in view.technologies.items()
            if any(
                wanted & set(item.version.domains)
                for item in technology.documents
            )
        }
        return "domain", sorted(domains), ids
    words = [word for word in _words(query) if len(word) >= 4]
    ids = {
        technology_id
        for technology_id, technology in view.technologies.items()
        if words
        and any(
            _word_match(word, part)
            for word in words
            for part in _words(technology.label)
        )
    }
    if ids:
        return "label", [], ids
    return "all", [], set(view.technologies)


def _trend(technology, cutoff: date, quarters: int) -> List[int]:
    series = [
        (day.observed, day.mentions)
        for item in technology.dated_documents
        for day in item.mentions
    ]
    counts = []
    for index in reversed(range(quarters)):
        end = months_before(cutoff, 3 * index)
        start = months_before(cutoff, 3 * (index + 1))
        counts.append(sum(n for when, n in series if start < when <= end))
    return counts


def _value(value: Any) -> str:
    if value is None:
        return "нет данных"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _sources(corpus, technology, tier_max: int, limit: int):
    documents = sorted(
        technology.documents,
        key=lambda item: (
            item.undated,
            -(item.version.version_date or date.min).toordinal(),
            item.document_id,
        ),
    )[:limit]
    result = []
    for item in documents:
        info = corpus.document_info.get(item.document_id, {})
        share = item.version.reliability_tier / tier_max
        result.append(
            {
                "title": info.get("title") or item.document_id,
                "url": info.get("url"),
                "date": (
                    item.version.version_date.isoformat()
                    if item.version.version_date
                    else "без даты"
                ),
                "type": DOCUMENT_TYPES.get(item.family, item.family),
                "lang": "—",
                "trust": (
                    "high"
                    if share >= 1
                    else "medium"
                    if share >= 0.5
                    else "low"
                ),
            }
        )
    return result


def _signal(corpus, view, item, config, domain_names) -> Dict[str, Any]:
    explanation = config["explanation"]
    rules = config["candidates"]
    row = item["row"]
    technology = view.technologies[item["technology_id"]]
    domains = Counter(
        domain_names[domain]
        for document in technology.documents
        for domain in document.version.domains
        if domain in domain_names
    )
    rank = row.get("max_maturity_rank")
    stage = STAGES.get(rank, "не определена")
    features = item["top_features"]
    sources = independent_sources(
        technology, months_before(view.cutoff, rules["window_months"])
    )
    return {
        "id": item["technology_id"],
        "title": item["technology"],
        "domain": domains.most_common(1)[0][0] if domains else "—",
        "score": item["score"],
        "stage": stage,
        "summary": " · ".join(
            f"{feature['label']}: {_value(feature['value'])}"
            for feature in features
        ),
        # Contribution = weight × z-score of the feature among candidates.
        "predictors": [
            {
                "name": feature["label"],
                "weight": round(feature["contribution"], 2),
                "value": _value(feature["value"]),
            }
            for feature in features
        ],
        "trend": _trend(
            technology, view.cutoff, int(explanation["trend_quarters"])
        ),
        "description": (
            f"Впервые встречается в данных {row.get('first_seen_date')}; "
            f"документов: {row.get('document_count')}, упоминаний: "
            f"{row.get('mention_count')}."
        ),
        "technologyDefinition": corpus.technology_profiles.get(
            technology.technology_id
        ),
        "advantages": [],
        "cases": [],
        "reports": [],
        "quotes": evidence_quotes(
            corpus, technology, int(explanation["max_quotes"])
        ),
        "whyWeak": (
            f"Прошла правило отбора на {view.cutoff.isoformat()}: возраст "
            f"{row.get('technology_age_days')} дн. (не больше "
            f"{rules['max_age_years']} лет), {sources} независимых "
            f"источника за {rules['window_months']} мес., стадия: {stage}."
        ),
        "confidenceReason": (
            "Скор — взвешенная сумма z-оценок признаков среди всех "
            "кандидатов (веса в ranking.json), сжатая в 0..1 логистической "
            f"функцией; это порядок, а не вероятность. Сумма: "
            f"{item['raw_score']:+.2f}."
        ),
        "sources": _sources(
            corpus,
            technology,
            int(load_catalog("dataset")["reliability_tier_max"]),
            int(explanation["max_sources"]),
        ),
    }


def search_response(
    corpus: TemporalCorpus,
    query: str,
    snapshot: date,
    config: Optional[Mapping[str, Any]] = None,
    ranking: Optional[Ranking] = None,
) -> Dict[str, Any]:
    """TOP-K weak signals at T for the query, in the frontend contract.

    ``ranking`` may be a cached :func:`rank_snapshot` result of the same
    corpus and date.
    """
    config = config or load_catalog("ranking")
    ranking = ranking or _rank_snapshot(corpus, snapshot, config)
    view = corpus.view(snapshot)
    scope, matched, ids = _scope(view, query)
    domain_names = {
        stable_id("domain", domain["name"]): domain["name"]
        for domain in load_catalog("sources")["domains"]
    }
    chosen = [
        item for item in ranking.candidates if item["technology_id"] in ids
    ]
    signals = [
        _signal(corpus, view, item, config, domain_names)
        for item in chosen[: int(config["top_k"])]
    ]
    rejected: List[Dict[str, str]] = [
        {
            "title": item["technology"],
            "category": item["category"],
            "reason": item["reason"],
        }
        for item in ranking.rejected
        if item["technology_id"] in ids
    ][: int(config["top_k"])]
    notes: Set[str] = set()
    if scope == "all":
        notes.add(
            "Запрос не совпал ни с доменом, ни с названием технологии: "
            "показан общий ТОП по всем кандидатам."
        )
    return {
        "query": query,
        "snapshot": snapshot.isoformat(),
        "demo": False,
        "scope": scope,
        "matched": matched,
        "note": " ".join(sorted(notes)) or None,
        "stats": {
            "sourcesProcessed": len(view.documents),
            "candidates": len(chosen),
            "confident": sum(item["score"] > 0.75 for item in chosen),
        },
        "signals": signals,
        "rejected": rejected,
    }


class SearchService:
    """Serves searches from one graph read per ``api.cache_seconds``.

    Reading the whole dated graph and ranking a snapshot are the slow
    parts; both run once per cache period (ranking once per date), off the
    event loop.
    """

    def __init__(
        self,
        read_data: Callable[[], Awaitable[Dict[str, Any]]],
        config: Optional[Mapping[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read = read_data
        self._config = config or load_catalog("ranking")
        self._clock = clock
        self._loaded_at: Optional[float] = None
        self._corpus: Optional[TemporalCorpus] = None
        self._rankings: Dict[date, Ranking] = {}
        self._lock: Optional[asyncio.Lock] = None

    async def search(
        self, query: str, snapshot: Optional[date] = None
    ) -> Dict[str, Any]:
        snapshot = snapshot or date.today()
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            ttl = float(self._config["api"]["cache_seconds"])
            if (
                self._loaded_at is None
                or self._clock() - self._loaded_at > ttl
            ):
                data = await self._read()
                self._corpus = await asyncio.to_thread(TemporalCorpus, data)
                self._rankings = {}
                self._loaded_at = self._clock()
            if snapshot not in self._rankings:
                self._rankings[snapshot] = await asyncio.to_thread(
                    _rank_snapshot, self._corpus, snapshot, self._config
                )
            corpus, ranking = self._corpus, self._rankings[snapshot]
        return await asyncio.to_thread(
            search_response, corpus, query, snapshot, self._config, ranking
        )
