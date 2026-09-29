"""Weak-signal cards: the analyst table on top of the ranked candidates.

The ranking (:mod:`.scoring`) orders single graph concepts, named as one
document wrote them. An analyst table names a niche, lists who works on
it, says why it is a weak signal, where it stands and how fast it moves.
This module builds such cards:

1. the ranked pool at T is clustered by name vectors, so near-duplicate
   concepts ("MCP server scanner", "MCP security scanning") make one card;
2. a dossier per cluster is read from the snapshot: definitions,
   organizations with their roles, market events, money, maturity
   evidence, accepted claims with quotes, documents by year;
3. the model names the niche and writes the reasons from the dossier only
   (``prompts/signal.txt``); companies it names must be in the dossier;
4. code computes what can be counted: the stage from maturity evidence,
   the trend from dated documents, the score (stage points + trend points)
   and the sources.

Without a provider, or when the model fails, a card is still built from
the dossier alone and marked ``llm: false``.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from collections import Counter
from datetime import date
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.config import load_catalog
from ..graph.features import months_before
from ..graph.temporal import (
    CODE,
    PACKAGE,
    PATENT,
    SCHOLARLY,
    DocumentTrace,
    TechnologyView,
    TemporalCorpus,
    independence_groups,
)
from .scoring import rank_snapshot

logger = logging.getLogger(__name__)

Row = Dict[str, Any]

FAMILIES = {
    SCHOLARLY: "научная публикация",
    CODE: "репозиторий кода",
    PACKAGE: "пакет",
    PATENT: "патент",
}
ROLES = {
    "DEVELOPED_BY": "разработчик",
    "USED_BY": "пользователь",
    "FUNDED_BY": "инвестор",
}
EVENTS = {
    "funding_round": "раунд финансирования",
    "stealth_exit": "выход из stealth",
    "product_launch": "запуск продукта",
    "acquisition": "поглощение",
    "partnership": "партнёрство",
    "contract": "контракт",
    "standard": "стандарт",
    "regulation": "регулирование",
    "open_source_release": "открытый релиз",
    "public_audit": "публичный аудит",
}
MARKET_EVENT = "reports_market_event"
AUTHOR_ROLE = "организация авторов материала"


def _normal(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def _clip(text: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class SignalAnswer(BaseModel):
    """The model's part of a card; everything countable is code's."""

    model_config = ConfigDict(extra="ignore")
    is_signal: bool = True
    reject_reason: str = ""
    title: str = ""
    area: str = ""
    companies: List[str] = Field(default_factory=list)
    why_weak: str = ""
    stage: Optional[
        Literal["concept", "prototype", "pilot", "early_adoption"]
    ] = None
    stage_note: str = ""
    trend_note: str = ""

    @field_validator("stage", mode="before")
    @classmethod
    def _known_stage(cls, value: Any) -> Any:
        # An unknown stage word must not cost the whole card.
        allowed = ("concept", "prototype", "pilot", "early_adoption")
        return value if value in allowed else None

    @field_validator("companies", mode="before")
    @classmethod
    def _company_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part for part in value.split(",") if part.strip()]
        return [str(item) for item in value or [] if item]

    @field_validator(
        "reject_reason",
        "title",
        "area",
        "why_weak",
        "stage_note",
        "trend_note",
        mode="before",
    )
    @classmethod
    def _text(cls, value: Any) -> Any:
        return "" if value is None else _clip(value, 1200)


# Clusters -------------------------------------------------------------


def cluster_candidates(
    candidates: List[Row],
    embeddings: Mapping[str, List[float]],
    min_cosine: float,
    max_members: int,
) -> List[List[Row]]:
    """Greedy clusters in ranking order: the best candidate not yet taken
    seeds a cluster and takes the next ones whose name vector is within
    ``min_cosine`` of the seed. Candidates without a vector stay alone.
    """
    import numpy as np

    ids = [item["technology_id"] for item in candidates]
    with_vector = [index for index, key in enumerate(ids) if key in embeddings]
    similarity = None
    if len(with_vector) > 1:
        sizes = {len(embeddings[ids[index]]) for index in with_vector}
        if len(sizes) == 1:
            matrix = np.array(
                [embeddings[ids[index]] for index in with_vector], dtype=float
            )
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1.0, norms)
            similarity = matrix @ matrix.T
    position = {index: row for row, index in enumerate(with_vector)}
    taken = set()
    clusters = []
    for seed in range(len(candidates)):
        if seed in taken:
            continue
        taken.add(seed)
        members = [candidates[seed]]
        if similarity is not None and seed in position:
            for other in range(seed + 1, len(candidates)):
                if len(members) >= max_members:
                    break
                if (
                    other not in taken
                    and other in position
                    and similarity[position[seed], position[other]]
                    >= min_cosine
                ):
                    taken.add(other)
                    members.append(candidates[other])
        clusters.append(members)
    return clusters


# Counted facts ---------------------------------------------------------


def _documents(views: Iterable[TechnologyView]) -> List[DocumentTrace]:
    """Documents of the cluster, once each, with the earliest visibility."""
    merged: Dict[str, DocumentTrace] = {}
    for view in views:
        for item in view.documents:
            known = merged.get(item.document_id)
            if known is None or item.first_visible < known.first_visible:
                merged[item.document_id] = item
    return sorted(
        merged.values(),
        key=lambda item: (item.first_visible, item.document_id),
    )


def _independent(documents: List[DocumentTrace], since: date) -> int:
    recent = [
        item
        for item in documents
        if not item.undated and item.first_visible > since
    ]
    groups = independence_groups(item.version for item in recent)
    return len(
        {
            groups[item.version.version_id] or f"document:{item.document_id}"
            for item in recent
        }
    )


def trend_of(
    documents: List[DocumentTrace], cutoff: date, settings: Mapping
) -> Dict[str, Any]:
    """Trend category from dated documents: the last window against the
    one before, with add-one smoothing so one document is not infinite
    growth.
    """
    window = int(settings["window_months"])
    start = months_before(cutoff, window)
    before = months_before(cutoff, 2 * window)
    dated = [item for item in documents if not item.undated]
    recent = sum(start < item.first_visible <= cutoff for item in dated)
    previous = sum(before < item.first_visible <= start for item in dated)
    ratio = (recent + 1) / (previous + 1)
    if (
        recent >= int(settings["min_fast_documents"])
        and ratio >= settings["fast_ratio"]
    ):
        category = "fast"
    elif recent and ratio >= settings["growth_ratio"]:
        category = "growing"
    elif ratio >= settings["stable_ratio"]:
        category = "stable"
    else:
        category = "declining"
    by_year = Counter(
        item.first_visible.year
        for item in dated
        if item.first_visible <= cutoff
    )
    return {
        "category": category,
        "label": settings["labels"][category],
        "points": int(settings["points"][category]),
        "window_months": window,
        "documents_last_window": recent,
        "documents_previous_window": previous,
        "documents_by_year": {
            str(year): by_year[year] for year in sorted(by_year)
        },
    }


def stage_of(
    views: Iterable[TechnologyView], settings: Mapping
) -> Optional[Dict[str, Any]]:
    """Card stage from maturity evidence visible at T: the highest stage,
    with the first one when the evidence moved up ("Прототип → Пилот").
    None without evidence.
    """
    ranks = settings["ranks"]
    events = sorted(
        (
            event
            for view in views
            for event in view.maturity
            if str(event.data.get("stage_rank")) in ranks
        ),
        key=lambda event: event.observed,
    )
    if not events:
        return None
    top = max(events, key=lambda event: int(event.data["stage_rank"]))
    first = ranks[str(events[0].data["stage_rank"])]
    stage = ranks[str(top.data["stage_rank"])]
    labels = settings["labels"]
    return {
        "stage": stage,
        "first": first,
        "label": (
            f"{labels[first]} → {labels[stage]}"
            if first != stage
            and settings["points"][first] < settings["points"][stage]
            else labels[stage]
        ),
        "history": [
            {
                "date": event.observed.isoformat(),
                "stage": ranks[str(event.data["stage_rank"])],
                "trl": event.data.get("trl"),
            }
            for event in events
        ],
    }


# Dossier ---------------------------------------------------------------


def _qualifiers(data: Mapping) -> Dict[str, Any]:
    value = data.get("qualifiers_json") or data.get("qualifiers") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _accepted(data: Mapping) -> bool:
    return data.get("status") in (None, "accepted") and data.get(
        "verification_status"
    ) in (None, "supported")


def _role_parties(data: Mapping, role: str) -> List[tuple]:
    """(name, kind) of the concepts in one role of an assertion."""
    return [
        (str(item[1]), item[2] if len(item) > 2 else None)
        for item in data.get("role_labels") or []
        if len(item) > 1 and item[0] == role and item[1]
    ]


def _role_labels(data: Mapping, role: str) -> List[str]:
    return [name for name, _ in _role_parties(data, role)]


def _organizations(
    corpus: TemporalCorpus,
    views: List[TechnologyView],
    documents: List[DocumentTrace],
) -> List[Dict[str, Any]]:
    """Named organizations with the roles the sources give them; document
    affiliations come last and only as authors of a material.
    """
    found: Dict[str, Dict[str, Any]] = {}

    def add(name: str, role: str, kind: Optional[str] = None) -> None:
        key = _normal(name)
        if not key:
            return
        item = found.setdefault(
            key, {"name": name, "kind": kind, "roles": Counter()}
        )
        item["kind"] = item["kind"] or kind
        item["roles"][role] += 1

    for view in views:
        for event in view.relations:
            role = ROLES.get(str(event.data.get("relation")))
            name = event.data.get("target_label")
            if role and name:
                labels = event.data.get("target_labels") or []
                kind = event.data.get("target_kind") or next(
                    (
                        label
                        for label in labels
                        if label in ("Company", "University")
                    ),
                    None,
                )
                add(str(name), role, kind)
        for event in view.assertions:
            data = event.data
            if not _accepted(data):
                continue
            if data.get("predicate") == MARKET_EVENT:
                what = EVENTS.get(str(_qualifiers(data).get("event")))
                for name, kind in _role_parties(data, "ORGANIZATION"):
                    add(name, what or "событие", kind)
            elif data.get("predicate") == "reported_investment":
                for name, kind in _role_parties(data, "ORGANIZATION"):
                    add(name, "получатель инвестиций", kind)
    for item in documents:
        version = item.version
        for kind, names in (
            ("Company", version.companies),
            ("University", version.universities),
            (None, version.organizations),
        ):
            for key in names:
                # Ids without a stored name are not shown.
                name = corpus.organization_names.get(str(key))
                if name:
                    add(name, AUTHOR_ROLE, kind)
    ranked = sorted(
        found.values(),
        key=lambda item: (
            set(item["roles"]) == {AUTHOR_ROLE},
            item["kind"] != "Company",
            -sum(item["roles"].values()),
            item["name"],
        ),
    )
    return [
        {
            "name": item["name"],
            "kind": item["kind"] or "Organization",
            "roles": sorted(item["roles"]),
        }
        for item in ranked
    ]


def _events(
    corpus: TemporalCorpus, views: List[TechnologyView], quote_chars: int
) -> List[Dict[str, Any]]:
    events = []
    seen = set()
    for view in views:
        for event in view.assertions:
            data = event.data
            if data.get("predicate") != MARKET_EVENT or not _accepted(data):
                continue
            key = data.get("assertion_id") or id(event)
            if key in seen:
                continue
            seen.add(key)
            qualifiers = _qualifiers(data)
            events.append(
                {
                    "date": event.observed.isoformat(),
                    "event": qualifiers.get("event"),
                    "round": qualifiers.get("round"),
                    "organizations": _role_labels(data, "ORGANIZATION"),
                    "modality": data.get("modality"),
                    "quote": _clip(data.get("quote"), quote_chars),
                    "document_id": _document_of(corpus, data),
                }
            )
        for event in view.economics:
            data = event.data
            if data.get("status") in ("rejected",) or not data.get(
                "amount_text"
            ):
                continue
            key = data.get("evidence_id") or id(event)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "date": event.observed.isoformat(),
                    "event": "money:" + str(data.get("category")),
                    "amount": data.get("amount_text"),
                    "currency": data.get("currency"),
                    "modality": data.get("modality"),
                    "document_id": _document_of(corpus, data),
                }
            )
    events.sort(key=lambda item: item["date"], reverse=True)
    return events


def _document_of(corpus: TemporalCorpus, data: Mapping) -> Optional[str]:
    version = corpus.versions.get(str(data.get("version_id") or ""))
    return version.document_id if version else None


def _claims(
    corpus: TemporalCorpus,
    views: List[TechnologyView],
    limit: int,
    quote_chars: int,
) -> List[Dict[str, Any]]:
    """Accepted claims other than market events, newest first, one quote
    per claim."""
    claims = []
    seen = set()
    for view in views:
        for event in view.assertions:
            data = event.data
            quote = data.get("quote")
            if (
                data.get("predicate") == MARKET_EVENT
                or not _accepted(data)
                or not quote
                or data.get("polarity") == "negated"
            ):
                continue
            key = (data.get("predicate"), _normal(str(quote)))
            if key in seen:
                continue
            seen.add(key)
            qualifiers = _qualifiers(data)
            claims.append(
                {
                    "date": event.observed.isoformat(),
                    "predicate": data.get("predicate"),
                    "modality": data.get("modality"),
                    **(
                        {"stage": qualifiers["stage"]}
                        if qualifiers.get("stage")
                        else {}
                    ),
                    "quote": _clip(quote, quote_chars),
                    "document_id": _document_of(corpus, data),
                }
            )
    claims.sort(key=lambda item: item["date"], reverse=True)
    return claims[:limit]


def build_dossier(
    corpus: TemporalCorpus,
    cutoff: date,
    members: List[Row],
    config: Mapping,
) -> Dict[str, Any]:
    """Everything the card may say about one cluster at T."""
    settings = config["signals"]
    limits = settings["dossier"]
    view = corpus.view(cutoff)
    views = [view.technologies[item["technology_id"]] for item in members]
    documents = _documents(views)
    window = int(config["candidates"]["window_months"])
    trend = trend_of(documents, cutoff, settings["trend"])
    stage = stage_of(views, settings["stages"])
    events = _events(corpus, views, int(limits["max_quote_chars"]))
    organizations = _organizations(corpus, views, documents)
    dated = [item for item in documents if not item.undated]
    return {
        "snapshot": cutoff.isoformat(),
        "technologies": [
            {
                "label": item.label,
                "kind": corpus.kinds.get(item.technology_id, "Technology"),
                **(
                    {"definition": corpus.definitions[item.technology_id]}
                    if item.technology_id in corpus.definitions
                    else {}
                ),
                "documents": len(item.documents),
            }
            for item in views
        ],
        "documents": len(documents),
        "documents_in_corpus": len(view.documents),
        "independent_sources": _independent(
            documents, months_before(cutoff, window)
        ),
        "independent_window_months": window,
        "first_seen": (dated[0].first_visible.isoformat() if dated else None),
        "source_types": dict(
            Counter(
                FAMILIES.get(item.family, item.version.document_type)
                for item in documents
            )
        ),
        "trend": trend,
        "stage_evidence": stage,
        "organizations": organizations[: int(limits["max_organizations"])],
        "events": events[: int(limits["max_events"])],
        "claims": _claims(
            corpus,
            views,
            int(limits["max_claims"]),
            int(limits["max_quote_chars"]),
        ),
        "recent_documents": [
            {
                "date": (
                    item.version.version_date.isoformat()
                    if item.version.version_date
                    else None
                ),
                "type": FAMILIES.get(item.family, item.version.document_type),
                "title": corpus.document_info.get(item.document_id, {}).get(
                    "title"
                ),
            }
            for item in sorted(
                documents,
                key=lambda item: item.first_visible,
                reverse=True,
            )[: int(limits["max_documents"])]
        ],
    }


# Cards -----------------------------------------------------------------


def _sources(
    corpus: TemporalCorpus,
    documents: List[DocumentTrace],
    dossier: Mapping,
    limit: int,
) -> List[Dict[str, Any]]:
    """Documents that carry events, then claims, then the newest."""
    weight = Counter()
    for item in dossier["events"]:
        weight[item.get("document_id")] += 2
    for item in dossier["claims"]:
        weight[item.get("document_id")] += 1
    chosen = sorted(
        documents,
        key=lambda item: (
            not corpus.document_info.get(item.document_id, {}).get("url"),
            -weight[item.document_id],
            -item.first_visible.toordinal(),
            item.document_id,
        ),
    )[:limit]
    result = []
    for item in chosen:
        info = corpus.document_info.get(item.document_id, {})
        result.append(
            {
                "title": info.get("title") or item.document_id,
                "url": info.get("url"),
                "date": (
                    item.version.version_date.isoformat()
                    if item.version.version_date
                    else None
                ),
            }
        )
    return result


def _companies(
    answer: Optional[SignalAnswer], dossier: Mapping, limit: int
) -> List[str]:
    """Names the model chose, kept only when they are in the dossier;
    the dossier's organizations with roles when none is left.
    """
    known = [item["name"] for item in dossier["organizations"]]
    keys = {_normal(name): name for name in known}
    chosen: List[str] = []
    seen = set()
    for text in answer.companies if answer else []:
        text = _clip(text, 200)
        base = _normal(text.split("(")[0])
        match = keys.get(base) or next(
            (
                name
                for key, name in keys.items()
                if len(base) >= 3
                and len(key) >= 3
                and (base in key or key in base)
            ),
            None,
        )
        if match is None or _normal(match) in seen:
            continue
        seen.add(_normal(match))
        chosen.append(text)
        if len(chosen) >= limit:
            break
    if chosen:
        return chosen
    explicit = [
        item["name"]
        for item in dossier["organizations"]
        if set(item["roles"]) != {AUTHOR_ROLE}
    ]
    rest = [
        item["name"]
        for item in dossier["organizations"]
        if item["name"] not in explicit
    ]
    return (explicit + rest)[:limit]


def _fallback_why(dossier: Mapping) -> str:
    parts = [
        f"Документов: {dossier['documents']} из "
        f"{dossier['documents_in_corpus']} в корпусе, независимых "
        f"источников за {dossier['independent_window_months']} мес.: "
        f"{dossier['independent_sources']}"
    ]
    if dossier["first_seen"]:
        parts.append(f"впервые встречается {dossier['first_seen']}")
    if dossier["source_types"]:
        parts.append(
            "типы источников: "
            + ", ".join(
                f"{name} ({count})"
                for name, count in sorted(dossier["source_types"].items())
            )
        )
    events = [
        EVENTS.get(str(item.get("event")), str(item.get("event")))
        for item in dossier["events"]
    ]
    if events:
        parts.append("события: " + ", ".join(sorted(set(events))))
    return "; ".join(parts) + "."


def _fallback_trend(trend: Mapping) -> str:
    years = ", ".join(
        f"{year}: {count}"
        for year, count in trend["documents_by_year"].items()
    )
    return (
        f"документов за последние {trend['window_months']} мес.: "
        f"{trend['documents_last_window']}, за предыдущие: "
        f"{trend['documents_previous_window']}"
        + (f" (по годам: {years})" if years else "")
    )


def make_card(
    corpus: TemporalCorpus,
    cutoff: date,
    members: List[Row],
    dossier: Mapping,
    answer: Optional[SignalAnswer],
    config: Mapping,
) -> Dict[str, Any]:
    """One table row: the model's words where it answered, code's counts
    everywhere."""
    settings = config["signals"]
    stages = settings["stages"]
    areas = settings["areas"]
    evidence = dossier["stage_evidence"]
    if evidence is not None:
        stage, stage_label = evidence["stage"], evidence["label"]
    elif answer is not None and answer.stage in stages["labels"]:
        stage = answer.stage
        stage_label = stages["labels"][stage]
    else:
        stage, stage_label = None, "Не определена"
    if answer is not None and answer.stage_note:
        stage_label += f" ({answer.stage_note.strip(' ()')})"
    trend = dossier["trend"]
    note = (answer.trend_note if answer else "") or _fallback_trend(trend)
    stage_points = int(stages["points"][stage]) if stage else 0
    view = corpus.view(cutoff)
    documents = _documents(
        view.technologies[item["technology_id"]] for item in members
    )
    area = answer.area if answer else ""
    return {
        "title": (answer.title if answer else "") or members[0]["technology"],
        "area": area if area in areas else areas[-1],
        "companies": _companies(
            answer, dossier, int(settings["max_companies"])
        ),
        "why_weak": (answer.why_weak if answer else "")
        or _fallback_why(dossier),
        "stage": stage,
        "stage_label": stage_label,
        "trend": trend["category"],
        "trend_label": f"{trend['label']} — {note}",
        "score": stage_points + trend["points"],
        "stage_points": stage_points,
        "trend_points": trend["points"],
        "sources": _sources(
            corpus, documents, dossier, int(settings["max_sources"])
        ),
        "technology_ids": [item["technology_id"] for item in members],
        "technologies": [item["technology"] for item in members],
        "ranking_score": members[0]["raw_score"],
        # Retrospective labels (docs/hgt-pipeline-2026-09-29.md, 4.5): empty
        # until computed after the horizon; the card never guesses them.
        "signal_36m": None,
        "trend_36m": None,
        "llm": answer is not None,
        **(
            {"llm_stage": answer.stage}
            if answer is not None and answer.stage != stage
            else {}
        ),
    }


def signal_config(
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The ranking config with the candidate rule of cards: early
    adoption (commercial deployment) still counts as a weak signal."""
    value = copy.deepcopy(dict(config or load_catalog("ranking")))
    value["candidates"]["max_stage_rank"] = int(
        value["signals"]["max_stage_rank"]
    )
    return value


async def _answer(
    provider, system: str, dossier: Mapping, config: Mapping
) -> Optional[SignalAnswer]:
    from ..llm.client import LLMError

    settings = config["signals"]
    try:
        return await provider.generate(
            SignalAnswer,
            system,
            {
                "areas": settings["areas"],
                "max_companies": settings["max_companies"],
                "dossier": dossier,
            },
            # A card is a short answer: the review budget fits it.
            stage="review",
        )
    except LLMError as exc:
        logger.warning(
            "Signal card without the model for %s: %s",
            dossier["technologies"][0]["label"],
            exc.code,
        )
        return None


async def build_cards(
    corpus: TemporalCorpus,
    snapshot: date,
    provider=None,
    config: Optional[Mapping[str, Any]] = None,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """The weak-signal table at T: cards ordered by score, then by the
    ranking. ``query`` narrows the pool like ``/api/search``.
    """
    from .search import _scope

    config = signal_config(config)
    settings = config["signals"]
    top_k = int(top_k or settings["top_k"])
    ranking = rank_snapshot(corpus, snapshot, config)
    view = corpus.view(snapshot)
    candidates = ranking.candidates
    scope = None
    if query:
        scope, _, ids = _scope(view, query)
        candidates = [
            item for item in candidates if item["technology_id"] in ids
        ]
    pool = candidates[: int(settings["pool"])]
    clusters = cluster_candidates(
        pool,
        corpus.embeddings_at(snapshot),
        float(settings["cluster"]["min_cosine"]),
        int(settings["cluster"]["max_members"]),
    )
    dossiers = [
        build_dossier(corpus, snapshot, members, config)
        for members in clusters
    ]
    answers: List[Optional[SignalAnswer]] = [None] * len(clusters)
    if provider is not None:
        from ..llm.pipeline import _prompt

        system = _prompt("signal")
        limit = asyncio.Semaphore(int(settings["concurrency"]))

        async def one(index: int) -> None:
            async with limit:
                answers[index] = await _answer(
                    provider, system, dossiers[index], config
                )

        await asyncio.gather(*(one(index) for index in range(len(clusters))))
    cards, rejected = [], []
    for members, dossier, answer in zip(clusters, dossiers, answers):
        if answer is not None and not answer.is_signal:
            rejected.append(
                {
                    "technologies": [item["technology"] for item in members],
                    "reason": answer.reject_reason or "не слабый сигнал",
                }
            )
            continue
        cards.append(
            make_card(corpus, snapshot, members, dossier, answer, config)
        )
    cards.sort(key=lambda card: (-card["score"], -card["ranking_score"]))
    cards = cards[:top_k]
    for number, card in enumerate(cards, 1):
        card["number"] = number
    return {
        "snapshot": snapshot.isoformat(),
        "query": query,
        "scope": scope,
        "areas": sorted({card["area"] for card in cards}),
        "stats": {
            "candidates": len(candidates),
            "pool": len(pool),
            "clusters": len(clusters),
            "cards": len(cards),
            "rejected_by_model": len(rejected),
            "without_model": sum(not card["llm"] for card in cards),
        },
        "cards": cards,
        "rejected": rejected,
    }
