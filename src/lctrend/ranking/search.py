"""Hybrid technology search: GigaChat cosine + BM25 + weak-signal score.

The weak-signal score is a provisional heuristic, not a model probability.
Only technologies visible in the dated graph snapshot may be returned.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
import unicodedata
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
    Sequence,
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
QUERY_STOPWORDS = frozenset(
    "в во на для по из и или the a an of for in on and or "
    "технология технологии решение решения перспективный перспективные "
    "слабый слабые сигнал сигналы тренд тренды новый новые область "
    "области сфера направление emerging technology technologies trend "
    "trends weak signals new solutions field".split()
)


class EmbeddingIndexError(ValueError):
    """Stored technology vectors cannot be queried with GigaChat."""


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", text.casefold().replace("ё", "е"))


def _search_words(text: str) -> List[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return [
        word
        for word in re.findall(r"[\w+#]+", text)
        if word not in QUERY_STOPWORDS
    ]


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
    """Lexical scope for reports and query-relevant rejection reasons."""
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


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if (
        len(left) != len(right)
        or not left
        or not all(math.isfinite(value) for value in (*left, *right))
    ):
        return 0.0
    a = math.sqrt(sum(value * value for value in left))
    b = math.sqrt(sum(value * value for value in right))
    if not a or not b:
        return 0.0
    result = sum(x * y for x, y in zip(left, right)) / (a * b)
    return max(-1.0, min(1.0, result))


def _hybrid_matches(corpus, view, candidates, query, query_embedding, config):
    """Rank matches; a high signal score cannot create relevance."""
    settings = config["search"]
    domain_catalog = load_catalog("sources")["domains"]
    domain_text = {
        stable_id("domain", domain["name"]): " ".join(
            [domain["name"], *domain.get("aliases", [])]
        )
        for domain in domain_catalog
    }
    documents = {}
    for item in candidates:
        technology_id = item["technology_id"]
        technology = view.technologies[technology_id]
        domains = {
            domain
            for document in technology.documents
            for domain in document.version.domains
        }
        # The name is more precise than a broad domain or source title.
        # Definitions are not versioned; using them at a historical T
        # would leak later text into the search result.
        text = " ".join(
            [technology.label] * 3
            + [domain_text.get(domain, "") for domain in sorted(domains)]
        )
        documents[technology_id] = Counter(_search_words(text))
    query_terms = set(_search_words(query))
    # A named domain can be inflected ("финтехе") or abbreviated in the
    # query; its canonical name is a safe lexical expansion.
    for name in match_domains(query):
        query_terms.update(_search_words(name))
    if not query_terms:
        return []
    document_frequency = Counter(
        word for tokens in documents.values() for word in tokens
    )
    size = len(documents)
    average_length = sum(
        map(lambda tokens: sum(tokens.values()), documents.values())
    ) / max(size, 1)
    k1, b = 1.2, 0.75
    lexical = {}
    for technology_id, tokens in documents.items():
        length = sum(tokens.values())
        score = 0.0
        for word in query_terms:
            frequency = tokens[word]
            if frequency:
                idf = math.log1p(
                    (size - document_frequency[word] + 0.5)
                    / (document_frequency[word] + 0.5)
                )
                score += (
                    idf
                    * frequency
                    * (k1 + 1)
                    / (frequency + k1 * (1 - b + b * length / average_length))
                )
        lexical[technology_id] = score
    lexical_max = max(lexical.values(), default=0.0)
    vectors = corpus.embeddings_at(view.cutoff)
    matches = []
    for item in candidates:
        technology_id = item["technology_id"]
        vector = vectors.get(technology_id)
        cosine = (
            _cosine(query_embedding, vector)
            if query_embedding and vector
            else 0.0
        )
        bm25 = lexical[technology_id]
        if bm25 <= 0 and cosine < settings["min_cosine"]:
            continue
        semantic = max(0.0, cosine)
        lexical_score = bm25 / lexical_max if lexical_max else 0.0
        available = (
            settings["cosine_weight"] if vector and query_embedding else 0
        ) + (settings["bm25_weight"] if bm25 else 0)
        relevance = (
            settings["cosine_weight"]
            * semantic
            * bool(vector and query_embedding)
            + settings["bm25_weight"] * lexical_score
        ) / available
        final = (
            settings["relevance_weight"] * relevance
            + settings["signal_weight"] * item["score"]
        )
        matches.append(
            {
                **item,
                "search_score": final,
                "relevance_score": relevance,
                "semantic_similarity": cosine
                if vector and query_embedding
                else None,
                "bm25_score": bm25,
            }
        )
    matches.sort(
        key=lambda item: (
            -item["search_score"],
            -item["relevance_score"],
            item["technology_id"],
        )
    )
    return matches


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
        "score": round(item["search_score"], 4),
        "relevanceScore": round(item["relevance_score"], 4),
        "weakSignalScore": round(item["score"], 4),
        "semanticSimilarity": (
            round(item["semantic_similarity"], 4)
            if item["semantic_similarity"] is not None
            else None
        ),
        "bm25Score": round(item["bm25_score"], 4),
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
            f"Итоговый скор {item['search_score']:.2f} = релевантность "
            f"{item['relevance_score']:.2f} × "
            f"{config['search']['relevance_weight']:.2f} + скор слабого "
            f"сигнала {item['score']:.2f} × "
            f"{config['search']['signal_weight']:.2f}. Последний рассчитан "
            "из признаков графа, а не обученной моделью; это не вероятность."
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
    query_embedding: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """TOP-K weak signals at T for the query, in the frontend contract.

    ``ranking`` may be a cached :func:`rank_snapshot` result of the same
    corpus and date.
    """
    config = config or load_catalog("ranking")
    ranking = ranking or _rank_snapshot(corpus, snapshot, config)
    view = corpus.view(snapshot)
    matched = sorted(match_domains(query))
    rejected_scope, _, rejected_ids = _scope(view, query)
    domain_names = {
        stable_id("domain", domain["name"]): domain["name"]
        for domain in load_catalog("sources")["domains"]
    }
    chosen = _hybrid_matches(
        corpus, view, ranking.candidates, query, query_embedding, config
    )
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
        if rejected_scope != "all" and item["technology_id"] in rejected_ids
    ][: int(config["top_k"])]
    notes: Set[str] = set()
    if not _search_words(query):
        notes.add("Уточните технологию или предметную область запроса.")
    elif not query_embedding:
        notes.add("Семантический поиск недоступен: использован только BM25.")
    if not chosen:
        notes.add("По запросу не найдено подходящих технологий.")
    return {
        "query": query,
        "snapshot": snapshot.isoformat(),
        "demo": False,
        "scope": "hybrid" if query_embedding else "lexical",
        "matched": matched,
        "note": " ".join(sorted(notes)) or None,
        "stats": {
            "sourcesProcessed": len(view.documents),
            "candidates": len(chosen),
            "confident": sum(
                item["score"] > 0.75 for item in chosen[: int(config["top_k"])]
            ),
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
        embed_query: Optional[
            Callable[[str, str], Awaitable[Sequence[float]]]
        ] = None,
    ) -> None:
        self._read = read_data
        self._config = config or load_catalog("ranking")
        self._clock = clock
        self._embed_query = embed_query
        self._embedder = None
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
        vector = None
        if (
            ranking.candidates
            and _search_words(query)
            and corpus.embeddings_at(snapshot)
        ):
            model = corpus.embedding_model
            if not model or not model.startswith("EmbeddingsGiga"):
                raise EmbeddingIndexError(
                    "Векторы технологий не совместимы с GigaChat: "
                    "перестройте эмбеддинги технологий"
                )
            if self._embed_query is not None:
                vector = await self._embed_query(query, model)
            else:
                if self._embedder is None:
                    from ..llm.client import JsonLLM

                    self._embedder = JsonLLM.from_environment(
                        provider="gigachat"
                    )
                vector = (await self._embedder.embed([query], model))[0]
            dimensions = len(next(iter(corpus.embeddings.values())))
            if len(vector) != dimensions:
                raise EmbeddingIndexError(
                    "Размерность эмбеддинга запроса не совпала с графом"
                )
        return await asyncio.to_thread(
            search_response,
            corpus,
            query,
            snapshot,
            self._config,
            ranking,
            vector,
        )
