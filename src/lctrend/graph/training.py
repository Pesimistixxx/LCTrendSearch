from __future__ import annotations

import csv
import json
import logging
import math
from bisect import bisect_left, bisect_right
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..core.config import load_catalog
from .temporal import (
    CODE,
    PACKAGE,
    PATENT,
    TemporalCorpus,
    independence_groups,
    parse_date,
)

logger = logging.getLogger(__name__)

FIELDNAMES = [
    "technology_id",
    "technology",
    "snapshot_date",
    "document_count",
    "mention_count",
    "documents_last_year",
    "country_count",
    "company_count",
    "university_count",
    "domain_count",
    "source_count",
    "task_count",
    "future_document_count",
    "label_realized_3y",
]
# From lctrend.taxonomy.taxonomy_features, built at the same snapshot.
TAXONOMY_FIELDS = [
    "taxonomy_level",
    "taxonomy_node_size",
    "taxonomy_sibling_count",
    "taxonomy_general_term",
    "semantic_novelty",
    "branch_growth",
    "branch_new_share",
    "new_branch_in_known_area",
]
FEATURE_FIELDS = [
    "technology_id",
    "technology",
    "snapshot_date",
    "first_seen_date",
    "technology_age_days",
    "document_count",
    "mention_count",
    "documents_last_year",
    "publication_growth",
    "citation_count",
    "country_count",
    "company_count",
    "university_count",
    "domain_count",
    "source_type_diversity",
    "independence_group_diversity",
    "task_count",
    "new_relation_count",
    "patent_data_available",
    "repository_data_available",
    "economic_data_available",
    "developer_count",
    "user_count",
    "funder_count",
    "text_country_count",
    "parent_technology_count",
    "max_maturity_rank",
    "max_trl",
    "economic_evidence_count",
    *TAXONOMY_FIELDS,
]

SIGNAL_COUNTS = {
    "DEVELOPED_BY": "developer_count",
    "USED_BY": "user_count",
    "FUNDED_BY": "funder_count",
    "DEVELOPED_IN": "text_country_count",
    "SUBTECHNOLOGY_OF": "parent_technology_count",
}


def _signal_features(
    rows: Iterable[Dict[str, object]], cutoff: date
) -> Dict[str, Dict[str, object]]:
    """Aggregate dated text signals known at the snapshot date."""
    targets: Dict[str, Dict[str, set]] = {}
    features: Dict[str, Dict[str, object]] = {}
    for row in rows:
        observed = row.get("observed_at")
        if not observed or _date(str(observed)) > cutoff:
            continue
        technology_id = str(row["technology_id"])
        item = features.setdefault(
            technology_id,
            {
                **{name: 0 for name in SIGNAL_COUNTS.values()},
                "max_maturity_rank": None,
                "max_trl": None,
                "economic_evidence_count": 0,
            },
        )
        signal = row["signal"]
        if signal in SIGNAL_COUNTS:
            targets.setdefault(technology_id, {}).setdefault(
                signal, set()
            ).add(row["target_id"])
            item[SIGNAL_COUNTS[signal]] = len(targets[technology_id][signal])
        elif signal == "MATURITY":
            for key, value in (
                ("max_maturity_rank", row.get("value")),
                ("max_trl", row.get("target_id")),
            ):
                if value is not None:
                    item[key] = max(int(value), item[key] or 0)
        elif signal == "ECONOMIC":
            item["economic_evidence_count"] += 1
    return features


def _date(value: str) -> date:
    return date.fromisoformat(value[:10])


def build_training_rows(
    mentions: Iterable[Dict[str, object]],
    documents: Iterable[Dict[str, object]],
    tasks: Iterable[Dict[str, object]],
    start_year: int,
    horizon_years: int,
    min_documents: int,
    positive_future_documents: int,
    negative_future_documents: int,
) -> List[Dict[str, object]]:
    document_data = {
        record["document_id"]: {
            **record,
            "date": _date(str(record["created_at"])),
        }
        for record in documents
        if record["created_at"]
    }
    technology_documents: Dict[str, Dict[str, Dict[str, object]]] = {}
    names: Dict[str, str] = {}
    for record in mentions:
        document = document_data.get(record["document_id"])
        if not document:
            continue
        technology_id = str(record["technology_id"])
        names[technology_id] = str(record["technology"])
        technology_documents.setdefault(technology_id, {})[
            str(record["document_id"])
        ] = {
            **document,
            "mentions": int(record["mentions"]),
        }
    technology_tasks: Dict[str, List[Dict[str, object]]] = {}
    for record in tasks:
        technology_tasks.setdefault(str(record["technology_id"]), []).append(
            record
        )

    latest_date = max(item["date"] for item in document_data.values())
    latest_snapshot_year = latest_date.year - horizon_years
    rows = []
    for snapshot_year in range(start_year, latest_snapshot_year + 1):
        snapshot = date(snapshot_year, 1, 1)
        horizon_end = date(snapshot_year + horizon_years, 1, 1)
        for technology_id, by_document in technology_documents.items():
            past = [
                item
                for item in by_document.values()
                if item["date"] <= snapshot
            ]
            future = [
                item
                for item in by_document.values()
                if snapshot < item["date"] <= horizon_end
            ]
            if len(past) < min_documents:
                continue
            future_count = len(future)
            if future_count >= positive_future_documents:
                label = 1
            elif future_count <= negative_future_documents:
                label = 0
            else:
                continue
            recent_start = date(snapshot_year - 1, 1, 1)
            task_count = len(
                {
                    item["task_id"]
                    for item in technology_tasks.get(technology_id, [])
                    if item["observed_at"]
                    and _date(str(item["observed_at"])) <= snapshot
                }
            )
            rows.append(
                {
                    "technology_id": technology_id,
                    "technology": names[technology_id],
                    "snapshot_date": snapshot.isoformat(),
                    "document_count": len(past),
                    "mention_count": sum(
                        int(item["mentions"]) for item in past
                    ),
                    "documents_last_year": sum(
                        item["date"] > recent_start for item in past
                    ),
                    "country_count": len(
                        {value for item in past for value in item["countries"]}
                    ),
                    "company_count": len(
                        {value for item in past for value in item["companies"]}
                    ),
                    "university_count": len(
                        {
                            value
                            for item in past
                            for value in item["universities"]
                        }
                    ),
                    "domain_count": len(
                        {value for item in past for value in item["domains"]}
                    ),
                    "source_count": len({item["source_id"] for item in past}),
                    "task_count": task_count,
                    "future_document_count": future_count,
                    "label_realized_3y": label,
                }
            )
    return rows


def write_training_rows(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(values)
    return len(values)


def build_feature_rows(
    mentions: Iterable[Dict[str, object]],
    documents: Iterable[Dict[str, object]],
    tasks: Iterable[Dict[str, object]],
    snapshot: str,
    signals: Iterable[Dict[str, object]] = (),
    taxonomy: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    cutoff = _date(snapshot)
    signal_features = _signal_features(signals, cutoff)
    empty_signals = {
        **{name: 0 for name in SIGNAL_COUNTS.values()},
        "max_maturity_rank": None,
        "max_trl": None,
        "economic_evidence_count": 0,
    }
    docs = {
        item["document_id"]: {**item, "date": _date(str(item["created_at"]))}
        for item in documents
        if item["created_at"]
    }
    by_technology: Dict[str, Dict[str, Dict[str, object]]] = {}
    names: Dict[str, str] = {}
    for item in mentions:
        document = docs.get(item["document_id"])
        if document and document["date"] <= cutoff:
            key = str(item["technology_id"])
            names[key] = str(item["technology"])
            by_technology.setdefault(key, {})[str(item["document_id"])] = {
                **document,
                "mentions": int(item["mentions"]),
            }
    task_map: Dict[str, set[str]] = {}
    for item in tasks:
        if item["observed_at"] and _date(str(item["observed_at"])) <= cutoff:
            task_map.setdefault(str(item["technology_id"]), set()).add(
                str(item["task_id"])
            )
    rows = []
    year_ago = date(cutoff.year - 1, cutoff.month, cutoff.day)
    two_years_ago = date(cutoff.year - 2, cutoff.month, cutoff.day)
    for technology_id, items in by_technology.items():
        values = list(items.values())
        recent = [item for item in values if item["date"] > year_ago]
        previous = [
            item for item in values if two_years_ago < item["date"] <= year_ago
        ]
        recent_relations = {
            (kind, value)
            for item in recent
            for kind, values_ in (
                ("country", item["countries"]),
                ("domain", item["domains"]),
            )
            for value in values_
        }
        prior_relations = {
            (kind, value)
            for item in values
            if item["date"] <= year_ago
            for kind, values_ in (
                ("country", item["countries"]),
                ("domain", item["domains"]),
            )
            for value in values_
        }
        metrics = [
            json.loads(str(item.get("metrics_json") or "{}"))
            for item in values
        ]
        families = {
            item.get("source_family")
            for item in values
            if item.get("source_family")
        }
        text_signals = signal_features.get(technology_id, empty_signals)
        rows.append(
            {
                "technology_id": technology_id,
                "technology": names[technology_id],
                "snapshot_date": cutoff.isoformat(),
                "first_seen_date": min(
                    item["date"] for item in values
                ).isoformat(),
                "technology_age_days": (
                    cutoff - min(item["date"] for item in values)
                ).days,
                "document_count": len(values),
                "mention_count": sum(item["mentions"] for item in values),
                "documents_last_year": len(recent),
                "publication_growth": len(recent) / max(1, len(previous)),
                "citation_count": sum(
                    float(item.get("citation_count", 0)) for item in metrics
                ),
                "country_count": len(
                    {x for item in values for x in item["countries"]}
                ),
                "company_count": len(
                    {x for item in values for x in item["companies"]}
                ),
                "university_count": len(
                    {x for item in values for x in item["universities"]}
                ),
                "domain_count": len(
                    {x for item in values for x in item["domains"]}
                ),
                "source_type_diversity": len(families),
                "independence_group_diversity": len(
                    {
                        item.get("independence_group")
                        for item in values
                        if item.get("independence_group")
                    }
                ),
                "task_count": len(task_map.get(technology_id, set())),
                "new_relation_count": len(recent_relations - prior_relations),
                "patent_data_available": "patent" in families,
                "repository_data_available": "code" in families,
                "economic_data_available": bool(
                    text_signals["economic_evidence_count"]
                ),
                **text_signals,
                **{
                    name: (taxonomy or {}).get(technology_id, {}).get(name)
                    for name in TAXONOMY_FIELDS
                },
            }
        )
    return rows


def write_feature_rows(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(values)
    return len(values)


# The functions above remain import-compatible for consumers of legacy files.
# Both CLI exports below use the same point-in-time feature builder.
IDENTITY_FIELDS = [
    "technology_id",
    "technology",
    "snapshot_id",
    "snapshot_date",
]
OUTCOME_FIELDS = [
    "horizon_end",
    "horizon_years",
    "future_document_count",
    "future_independent_sources",
    "future_patents",
    "future_repositories",
    "future_packages",
    "future_users",
    "future_commercial_evidence",
    "first_patent",
    "first_repository",
    "first_package",
    "first_user",
    "future_source_coverage_count",
    "outcome_observation_complete",
    "label_realized",
    "label_reason",
    # Retrospective labels (docs/hgt-pipeline-2026-09-29.md, 4.5): a real
    # early signal at T; a trend confirmed within 36 months after T. Empty
    # until computed from global series, never from the feature rules.
    "signal_36m",
    "trend_36m",
    "split",
]


def _cutoff(value: object) -> date:
    parsed = parse_date(value)
    if parsed is None:
        raise ValueError(f"Invalid snapshot date: {value!r}")
    return parsed


def _snapshot_features(snapshot, view, config):
    from . import features as f

    cutoff = snapshot.cutoff
    stages = config["stage_ranks"]
    patent_families = getattr(snapshot, "_patent_families", None)
    if patent_families is None:
        patent_families = {}
        for item in snapshot.documents.values():
            family = item.metadata.get("family_id")
            if family:
                patent_families[family] = patent_families.get(family, 0) + 1
        snapshot._patent_families = patent_families
    economic = f.economic_features(
        view,
        cutoff,
        config["factual_modalities"],
        config["reliability_tier_max"],
    )
    relations = f.relation_features(view)
    covered = snapshot.corpus.covered_families(cutoff, view.technology_id)
    confirmed_volume = (
        sum(item.family in (CODE, PACKAGE, PATENT) for item in view.documents)
        + relations["commercial_user_count"]
        + economic["economic_evidence_count"]
    )
    row = {
        **f.activity_features(view, cutoff),
        **f.dynamics_features(view, cutoff, config["growth_source_families"]),
        **f.convergence_features(view, cutoff, f.company_use_dates(view)),
        **f.participant_features(view, cutoff),
        **relations,
        **f.credibility_features(
            view,
            config["reliability_tier_max"],
            config["speculative_modalities"],
            confirmed_volume,
        ),
        **f.claim_slot_features(view, config["factual_modalities"]),
        **economic,
        **f.economic_layer_features(view, cutoff, covered),
        **f.maturity_features(view, snapshot, stages, patent_families),
        **f.quality_features(view),
        **f.coverage_features(
            view,
            covered,
            config["coverage_families"],
        ),
        "source_count": len(
            {item.version.source_id for item in view.documents}
        ),
    }
    row["growth_plateau_score"] = f.growth_plateau_score(
        row["mention_count"],
        row["mention_growth_12m"],
    )

    def targets(technology):
        if technology is None:
            return set()
        edges = {
            (event.data.get("relation"), event.data.get("target_id"))
            for event in technology.relations
        }
        for document in technology.documents:
            for kind in ("countries", "domains", "companies", "universities"):
                edges.update(
                    (kind, target)
                    for target in getattr(document.version, kind)
                )
        return edges

    prior = snapshot.corpus.view(f.months_before(cutoff, 12))
    row["new_relation_count"] = len(
        targets(view) - targets(prior.technologies.get(view.technology_id)),
    )
    return row


def _feature_schema(config):
    from .novelty import NOVELTY_FIELDS
    from .temporal import TechnologyView

    corpus = TemporalCorpus({})
    snapshot = corpus.view(date(2000, 1, 1))
    return (
        list(
            _snapshot_features(
                snapshot,
                TechnologyView("", ""),
                config,
            )
        )
        + list(NOVELTY_FIELDS)
        + [
            "feature_missingness_count",
            "data_completeness_score",
        ]
        + [
            f"{column}_snapshot_pct"
            for column in config.get("snapshot_percentiles", [])
        ]
    )


def dataset_feature_fields():
    """Predictor schema shared by CSV exports and root subgraph features."""
    return _feature_schema(load_catalog("dataset"))


def build_snapshot_rows(
    corpus: TemporalCorpus,
    snapshot,
    min_documents: int = 1,
    include_novelty: bool = True,
) -> List[Dict[str, object]]:
    """All predictors at T, shared by training and inference exports."""
    from .novelty import NOVELTY_FIELDS, novelty_features

    if min_documents < 1:
        raise ValueError("min_documents must be positive")
    config = load_catalog("dataset")
    current = corpus.view(_cutoff(snapshot))
    novelty = novelty_features(current, config) if include_novelty else {}
    rows = []
    for technology_id, view in sorted(current.technologies.items()):
        if len(view.documents) < min_documents:
            continue
        features = _snapshot_features(current, view, config)
        features.update(
            {
                name: novelty.get(technology_id, {}).get(name)
                for name in NOVELTY_FIELDS
            }
        )
        # Missingness covers predictor values, not dates or missingness flags.
        values = [
            value
            for name, value in features.items()
            if name != "first_seen_date"
            and not name.endswith(("_missing", "_covered"))
        ]
        missing = sum(
            value is None
            or (isinstance(value, float) and not math.isfinite(value))
            for value in values
        )
        features["feature_missingness_count"] = missing
        features["data_completeness_score"] = (
            1.0 - missing / len(values) if values else None
        )
        rows.append(
            {
                "technology_id": technology_id,
                "technology": view.label,
                "snapshot_id": current.cutoff.isoformat(),
                "snapshot_date": current.cutoff.isoformat(),
                **features,
            }
        )
    _snapshot_percentiles(rows, config.get("snapshot_percentiles", []))
    return rows


def _snapshot_percentiles(rows: List[Dict[str, object]], columns) -> None:
    """Rank of a value among the technologies of the same snapshot.

    Constant dollars remove inflation, not the growth of a whole field:
    a percentile within T compares a technology with its contemporaries,
    which carries over to snapshots the model has never seen.
    """
    for column in columns:
        values = sorted(
            float(row[column])
            for row in rows
            if isinstance(row.get(column), (int, float))
            and not isinstance(row.get(column), bool)
        )
        for row in rows:
            value = row.get(column)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                row[f"{column}_snapshot_pct"] = None
                continue
            below = bisect_left(values, float(value))
            equal = bisect_right(values, float(value)) - below
            row[f"{column}_snapshot_pct"] = (below + 0.5 * equal) / len(values)


def _future_outcomes(
    corpus, technology_id, snapshot, horizon_end, config, observed_by
):
    from .features import classify_economic, is_company

    before = corpus.view(snapshot).technologies[technology_id]
    after = corpus.view(horizon_end).technologies.get(technology_id, before)
    future = [
        item
        for item in after.dated_documents
        if snapshot < item.first_visible <= horizon_end
    ]

    previous_users = {
        event.data.get("target_id")
        for event in before.relations
        if event.data.get("relation") == "USED_BY" and is_company(event.data)
    }
    uses = [
        event
        for event in after.relations
        if snapshot < event.observed <= horizon_end
        and event.data.get("relation") == "USED_BY"
        and is_company(event.data)
        and event.data.get("target_id") not in previous_users
    ]
    economics = [
        event
        for event in after.economics
        if snapshot < event.observed <= horizon_end
        and event.data.get("category")
        in config["commercial_economic_categories"]
        and classify_economic(event.data, config["factual_modalities"])
        in ("fact", "reviewed")
    ]
    # Confirming documents joined through shared participants are one
    # source; documents naming nobody cannot prove independence.
    confirming = [item.version for item in future] + [
        version
        for event in uses + economics
        if (version := corpus.versions.get(event.data.get("version_id")))
        is not None
    ]
    groups = set(independence_groups(confirming).values())
    groups.discard(None)
    # Searches are known by the observation moment, usually long after the
    # horizon; each must still cover the whole horizon interval.
    coverage = corpus.covered_families(
        observed_by,
        technology_id,
        period_start=snapshot,
        period_end=horizon_end,
    )
    required = set(
        config["label"].get(
            "required_negative_families",
            config["coverage_families"],
        )
    )
    # Realization is the first artifact of its kind after T: a repository
    # added to earlier ones continues an implementation, it is not one.
    before_families = {item.family for item in before.documents}
    future_families = {item.family for item in future}

    def first(family):
        return family in future_families and family not in before_families

    return {
        "future_document_count": len(future),
        "future_independent_sources": len(groups),
        "future_patents": sum(item.family == PATENT for item in future),
        "future_repositories": sum(item.family == CODE for item in future),
        "future_packages": sum(item.family == PACKAGE for item in future),
        "future_users": len({event.data.get("target_id") for event in uses}),
        "future_commercial_evidence": len(economics),
        "first_patent": first(PATENT),
        "first_repository": first(CODE),
        "first_package": first(PACKAGE),
        "first_user": bool(uses) and not previous_users,
        "future_source_coverage_count": len(required & coverage),
        "outcome_observation_complete": required <= coverage,
    }


def _label(row, end_date, config):
    # Unknown maturity is a flagged feature (max_maturity_rank_missing),
    # not a reason to drop the label.
    rank = row["max_maturity_rank"]
    if rank is not None and rank >= config["label"]["commercial_stage_rank"]:
        return None, "already_commercial"
    if _cutoff(row["horizon_end"]) > end_date:
        return None, "horizon_censored"
    if row["future_independent_sources"] >= config["label"][
        "min_future_independent_sources"
    ] and any(
        row[name] for name in config["label"]["implementation_outcomes"]
    ):
        return 1, "realization_observed"
    if not row["outcome_observation_complete"]:
        return None, "source_coverage_incomplete"
    return 0, "no_realization_observed"


def temporal_split(rows, valid_snapshots=1, test_snapshots=1):
    """Assign ordered splits and purge overlapping target horizons.

    Unknown labels stay outside training. Outcomes of (T, T+H] ending on
    a later split's first snapshot use nothing after it and stay. A row
    whose horizon reaches past that snapshot is kept for audit with
    split='purged'.
    """
    if valid_snapshots < 0 or test_snapshots < 0:
        raise ValueError("split snapshot counts cannot be negative")
    dates = sorted(
        {
            row["snapshot_date"]
            for row in rows
            if row["label_realized"] is not None
        }
    )
    test_dates = set(dates[-test_snapshots:]) if test_snapshots else set()
    rest = dates[:-test_snapshots] if test_snapshots else dates
    if test_dates:
        rest = [
            when
            for when in rest
            if all(
                row["horizon_end"] <= min(test_dates)
                for row in rows
                if row["snapshot_date"] == when
                and row["label_realized"] is not None
            )
        ]
    valid_dates = set(rest[-valid_snapshots:]) if valid_snapshots else set()
    for row in rows:
        current = row["snapshot_date"]
        if row["label_realized"] is None:
            row["split"] = "unlabeled"
        elif current in test_dates:
            row["split"] = "test"
        elif current in valid_dates:
            row["split"] = "valid"
        else:
            boundary = min(valid_dates or test_dates, default=None)
            row["split"] = (
                "purged"
                if boundary and row["horizon_end"] > boundary
                else "train"
            )
    return rows


def build_dataset_rows(
    corpus: TemporalCorpus,
    start_year: int = 2015,
    horizon_years: int = 3,
    min_documents: int = 2,
    end_date=None,
    include_novelty: bool = True,
    model_grid: bool = False,
) -> List[Dict[str, object]]:
    """technology x snapshot -> predictors at T + outcomes in (T, T+H]."""
    if horizon_years < 1 or min_documents < 1:
        raise ValueError("horizon_years and min_documents must be positive")
    if start_year < 1 or start_year + horizon_years > 9999:
        raise ValueError("snapshot and horizon must be valid calendar years")
    config = load_catalog("dataset")
    if corpus.latest_date is None:
        return []
    end = _cutoff(end_date) if end_date is not None else corpus.latest_date
    # A requested end cannot make an unobserved horizon complete.
    end = min(end, corpus.latest_date)
    settings = config["snapshots"]
    step = int(settings["step_months"])
    if step <= 0:
        raise ValueError("snapshot step_months must be positive")
    if model_grid:
        from ..modeling.annotations import snapshot_grid

        snapshots = snapshot_grid(end, start_year)
    else:
        from calendar import monthrange

        def regular_snapshots():
            current = date(start_year, settings["month"], settings["day"])
            while current <= end:
                yield current
                index = current.year * 12 + current.month - 1 + step
                year, month = divmod(index, 12)
                if year > 9999:
                    break
                current = date(
                    year,
                    month + 1,
                    min(settings["day"], monthrange(year, month + 1)[1]),
                )

        snapshots = regular_snapshots()
    rows = []
    for current in snapshots:
        if current.year + horizon_years > 9999:
            break
        from calendar import monthrange

        horizon_end = date(
            current.year + horizon_years,
            current.month,
            min(
                current.day,
                monthrange(
                    current.year + horizon_years,
                    current.month,
                )[1],
            ),
        )
        for row in build_snapshot_rows(
            corpus,
            current,
            min_documents,
            include_novelty,
        ):
            row.update(
                {
                    "horizon_end": horizon_end.isoformat(),
                    "horizon_years": horizon_years,
                    **_future_outcomes(
                        corpus,
                        row["technology_id"],
                        current,
                        min(horizon_end, end),
                        config,
                        end,
                    ),
                }
            )
            if horizon_end > end:
                row["outcome_observation_complete"] = False
            row["label_realized"], row["label_reason"] = _label(
                row, end, config
            )
            row["signal_36m"] = row["trend_36m"] = None
            rows.append(row)
    return temporal_split(
        rows,
        config["split"]["valid_snapshots"],
        config["split"]["test_snapshots"],
    )


SPLITS = ("train", "valid", "test", "purged")


def _label_summary(values):
    """Class balance per split, label reasons and one-class warnings."""
    labelled = [row for row in values if row.get("label_realized") in (0, 1)]

    def balance(part):
        return {
            str(label): sum(row["label_realized"] == label for row in part)
            for label in (0, 1)
        }

    classes = {
        name: balance([row for row in labelled if row.get("split") == name])
        for name in SPLITS
    }
    classes["all"] = balance(labelled)
    reasons: Dict[str, int] = {}
    for row in values:
        reason = str(row.get("label_reason"))
        reasons[reason] = reasons.get(reason, 0) + 1
    warnings = []
    present = [label for label, count in classes["all"].items() if count]
    if len(present) < 2:
        warnings.append(
            "labels have one class only "
            f"({', '.join(present) or 'none'}): a classifier cannot be "
            "trained; see label_reasons"
        )
    for warning in warnings:
        logger.warning("Training set: %s", warning)
    return classes, dict(sorted(reasons.items())), warnings


def _write_temporal_rows(path, rows, training, corpus=None):
    values = list(rows)
    config = load_catalog("dataset")
    features = _feature_schema(config)
    fields = IDENTITY_FIELDS + features + (OUTCOME_FIELDS if training else [])
    fields = list(dict.fromkeys(fields))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    from .novelty import DUPLICATE_COLUMNS

    manifest = {
        "schema_version": config["schema_version"],
        "rows": len(values),
        "feature_columns": features,
        # Dropped duplicate columns and the feature column that replaces
        # each.
        "column_aliases": DUPLICATE_COLUMNS,
        "identity_columns": IDENTITY_FIELDS,
        "outcome_columns": OUTCOME_FIELDS if training else [],
        "missing_value": "empty CSV cell",
        "label_column": "label_realized" if training else None,
        # Strict mode: content waited for its collection and extraction.
        "as_known": bool(corpus is not None and corpus.as_known),
        # Visible in totals, excluded from dynamics and first_seen.
        "undated_documents": (
            corpus.undated_documents if corpus is not None else None
        ),
        "config": config,
        "splits": {
            name: sum(row.get("split") == name for row in values)
            for name in (*SPLITS, "unlabeled")
        }
        if training
        else {},
    }
    if training:
        (
            manifest["class_balance"],
            manifest["label_reasons"],
            manifest["warnings"],
        ) = _label_summary(values)
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(values)


def write_dataset_rows(
    path: Path,
    rows: Iterable[Dict[str, object]],
    corpus: Optional[TemporalCorpus] = None,
) -> int:
    return _write_temporal_rows(path, rows, True, corpus)


def write_snapshot_rows(
    path: Path,
    rows: Iterable[Dict[str, object]],
    corpus: Optional[TemporalCorpus] = None,
) -> int:
    return _write_temporal_rows(path, rows, False, corpus)
