from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

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
