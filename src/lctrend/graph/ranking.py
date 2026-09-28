"""Transparent TOP-15 of weak technology signals at a snapshot date T.

Until the temporal dataset has both label classes, a trained classifier is
not possible. This module ranks instead:

1. a rule on data known at T selects candidates (``ranking.json →
   candidates``); every other technology gets a rejection reason;
2. each weighted column of ``export-features`` is z-scored over all
   candidates at T; the score is the sum of ``weight × z``, so the
   contribution of a feature is exactly ``weight × z`` (no SHAP needed);
3. the explanation lists the largest contributions and quotes accepted
   assertions (document, date, URL) already stored in the graph.

The backtest checks the ranking honestly: TOP-K at T from data up to T,
then the share that grew in (T, T+H] against random samples of the same
candidate pool (precision@K).
"""

from __future__ import annotations

import math
import random
import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date
from statistics import fmean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.config import load_catalog
from .features import months_before
from .temporal import TechnologyView, TemporalCorpus, independence_groups

Row = Dict[str, Any]


def _normal(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def independent_sources(view: TechnologyView, since: date) -> int:
    """Independent teams among dated documents first seen after ``since``.

    Documents sharing a participant are one source; a document naming
    nobody counts as its own.
    """
    recent = [
        item for item in view.dated_documents if item.first_visible > since
    ]
    groups = independence_groups(item.version for item in recent)
    return len(
        {
            groups[item.version.version_id] or f"document:{item.document_id}"
            for item in recent
        }
    )


def rejection(
    row: Row,
    view: TechnologyView,
    kind: str,
    cutoff: date,
    rules: Mapping[str, Any],
) -> Optional[Tuple[str, str]]:
    """``(category, reason)`` when the rule at T rejects a technology."""
    if kind not in rules["kinds"]:
        return "noise", f"вид {kind} не рассматривается"
    generic = {_normal(label) for label in rules["generic_labels"]}
    if _normal(view.label) in generic or row.get("taxonomy_general_term"):
        return "standard", "общий термин, а не конкретная технология"
    rank = row.get("max_maturity_rank")
    if rank is not None and rank > rules["max_stage_rank"]:
        return "mature", "стадия выше пилота: уже внедряется"
    age = row.get("technology_age_days")
    if age is None:
        return "noise", "нет датированных документов"
    if age > rules["max_age_years"] * 365.25:
        return "mature", (
            f"известна с {row['first_seen_date']}: старше "
            f"{rules['max_age_years']} лет"
        )
    sources = independent_sources(
        view, months_before(cutoff, rules["window_months"])
    )
    if sources < rules["min_independent_sources"]:
        return "noise", (
            f"{sources} независимых источника за "
            f"{rules['window_months']} мес. (нужно "
            f"≥{rules['min_independent_sources']})"
        )
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def score_rows(
    rows: Sequence[Row], weights: Mapping[str, float], z_clip: float
) -> List[Row]:
    """Weighted sum of z-scores, highest first.

    z is computed over the given rows; a missing value is neutral (z = 0),
    a constant column contributes nothing. ``score`` maps the raw sum to
    0..1 with a logistic curve; it orders, it is not a probability.
    """
    stats = {}
    for name in weights:
        values = [
            value
            for row in rows
            if (value := _number(row.get(name))) is not None
        ]
        spread = pstdev(values) if len(values) > 1 else 0.0
        stats[name] = (fmean(values) if values else 0.0, spread)
    scored = []
    for row in rows:
        contributions = {}
        for name, weight in weights.items():
            value = _number(row.get(name))
            mean, spread = stats[name]
            z = 0.0 if value is None or not spread else (value - mean) / spread
            z = max(-z_clip, min(z_clip, z))
            contributions[name] = round(weight * z, 6) + 0.0
        raw = round(sum(contributions.values()), 6) + 0.0
        scored.append(
            {
                **row,
                "contributions": contributions,
                "raw_score": raw,
                "score": 1.0 / (1.0 + math.exp(-raw)),
            }
        )
    scored.sort(key=lambda item: (-item["raw_score"], item["technology_id"]))
    return scored


@dataclass
class Ranking:
    cutoff: date
    candidates: List[Row] = field(default_factory=list)
    rejected: List[Row] = field(default_factory=list)


def rank_snapshot(
    corpus: TemporalCorpus,
    snapshot: date,
    config: Optional[Mapping[str, Any]] = None,
) -> Ranking:
    """Candidates at T ranked by the configured score, plus rejections."""
    from .training import build_snapshot_rows

    config = config or load_catalog("ranking")
    rules = config["candidates"]
    view = corpus.view(snapshot)
    rows = build_snapshot_rows(
        corpus, snapshot, include_novelty=bool(config["include_novelty"])
    )
    ranking = Ranking(snapshot)
    candidates = []
    for row in rows:
        technology = view.technologies[row["technology_id"]]
        kind = corpus.kinds.get(row["technology_id"], "Technology")
        rejected = rejection(row, technology, kind, snapshot, rules)
        if rejected is None:
            candidates.append(row)
        else:
            ranking.rejected.append(
                {
                    "technology_id": row["technology_id"],
                    "technology": row["technology"],
                    "category": rejected[0],
                    "reason": rejected[1],
                    "mention_count": row.get("mention_count") or 0,
                }
            )
    settings = config["score"]
    explanation = config["explanation"]
    for item in score_rows(
        candidates, settings["weights"], float(settings["z_clip"])
    ):
        top = sorted(
            item["contributions"].items(),
            key=lambda pair: (-abs(pair[1]), pair[0]),
        )[: int(explanation["top_features"])]
        ranking.candidates.append(
            {
                "technology_id": item["technology_id"],
                "technology": item["technology"],
                "raw_score": item["raw_score"],
                "score": item["score"],
                "contributions": item["contributions"],
                "top_features": [
                    {
                        "feature": name,
                        "label": settings["labels"].get(name, name),
                        "contribution": value,
                        "value": item.get(name),
                    }
                    for name, value in top
                ],
                "row": {
                    key: value
                    for key, value in item.items()
                    if key not in ("contributions", "raw_score", "score")
                },
            }
        )
    ranking.rejected.sort(
        key=lambda item: (-item["mention_count"], item["technology_id"])
    )
    return ranking


def evidence_quotes(
    corpus: TemporalCorpus, view: TechnologyView, limit: int
) -> List[Dict[str, Optional[str]]]:
    """Up to ``limit`` accepted assertions visible at T, newest first, one
    per document: the quote, the document title, its date and URL.
    """
    quotes, documents = [], set()
    for event in sorted(
        view.assertions,
        key=lambda item: (
            -item.observed.toordinal(),
            str(item.data.get("assertion_id")),
        ),
    ):
        data = event.data
        version = corpus.versions.get(str(data.get("version_id") or ""))
        if (
            data.get("status") != "accepted"
            or not data.get("quote")
            or version is None
            or version.document_id in documents
        ):
            continue
        documents.add(version.document_id)
        info = corpus.document_info.get(version.document_id, {})
        when = version.version_date or event.observed
        quotes.append(
            {
                "text": str(data["quote"]),
                "title": info.get("title"),
                "date": when.isoformat(),
                "url": info.get("url"),
            }
        )
        if len(quotes) == limit:
            break
    return quotes


def _years_after(value: date, years: int) -> date:
    year = value.year + years
    return date(
        year, value.month, min(value.day, monthrange(year, value.month)[1])
    )


def _documents_between(view, start: date, end: date) -> int:
    if view is None:
        return 0
    return sum(
        start < item.first_visible <= end for item in view.dated_documents
    )


def backtest(
    corpus: TemporalCorpus,
    snapshot: date,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """precision@K of the TOP-K at T against random samples of candidates.

    The ranking reads only data up to T. A candidate grew when its dated
    documents in (T, T+H] exceed ``min_growth_ratio`` times those in the
    preceding window of the same length and reach ``min_future_documents``.
    """
    config = config or load_catalog("ranking")
    settings = config["backtest"]
    horizon = int(settings["horizon_years"])
    horizon_end = _years_after(snapshot, horizon)
    ranking = rank_snapshot(corpus, snapshot, config)
    after = corpus.view(horizon_end).technologies
    before = corpus.view(snapshot).technologies
    past_start = _years_after(snapshot, -horizon)

    def outcome(item):
        technology_id = item["technology_id"]
        past = _documents_between(
            before.get(technology_id), past_start, snapshot
        )
        future = _documents_between(
            after.get(technology_id), snapshot, horizon_end
        )
        grew = (
            future >= int(settings["min_future_documents"])
            and future > float(settings["min_growth_ratio"]) * past
        )
        return {
            "technology_id": technology_id,
            "technology": item["technology"],
            "score": item["score"],
            "past_documents": past,
            "future_documents": future,
            "grew": grew,
        }

    outcomes = [outcome(item) for item in ranking.candidates]
    k = min(int(config["top_k"]), len(outcomes))
    top = outcomes[:k]
    grown = [item["grew"] for item in outcomes]
    precision = sum(item["grew"] for item in top) / k if k else None
    trials = int(settings["random_trials"])
    generator = random.Random(settings["seed"])
    samples = (
        [sum(generator.sample(grown, k)) / k for _ in range(trials)]
        if k
        else []
    )
    complete = (
        corpus.latest_date is not None and horizon_end <= corpus.latest_date
    )
    warnings = []
    if not complete:
        warnings.append(
            f"horizon ends {horizon_end.isoformat()} after the last data "
            f"({corpus.latest_date}): future growth is not fully observed"
        )
    if not k:
        warnings.append("no candidates at the snapshot")
    return {
        "snapshot": snapshot.isoformat(),
        "horizon_years": horizon,
        "horizon_end": horizon_end.isoformat(),
        "horizon_complete": complete,
        "candidates": len(outcomes),
        "k": k,
        "precision_at_k": precision,
        "base_rate": sum(grown) / len(grown) if grown else None,
        "random": {
            "trials": trials,
            "seed": settings["seed"],
            "mean_precision": fmean(samples) if samples else None,
            # Share of random samples at least as good as the TOP-K.
            "p_value": (
                sum(value >= precision for value in samples) / len(samples)
                if samples
                else None
            ),
        },
        "top": top,
        "warnings": warnings,
    }
