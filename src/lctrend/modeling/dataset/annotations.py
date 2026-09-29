"""Create a point-in-time pilot for two independent expert reviewers.

This is a review queue, not a source of automatic ground-truth labels.  In
particular, no absence of a patent or repository is interpreted as failure.
"""

from __future__ import annotations

import csv
import json
from calendar import monthrange
from collections import defaultdict
from datetime import date
from pathlib import Path

from ...graph.temporal import TemporalCorpus

PILOT_FIELDS = (
    "technology_id",
    "technology",
    "family_id",
    "snapshot_date",
    "horizon_12m_end",
    "horizon_end",
    "calendar_complete_12m",
    "calendar_complete_36m",
    "document_count",
    "first_seen_date",
    "source_families",
    "organizations",
    "recent_document_ids",
    "recent_documents",
    "reviewer_state",
    "reviewer_signal_12m",
    "reviewer_trend_12m",
    "reviewer_signal_36m",
    "reviewer_trend_36m",
    "evidence_notes",
)


def snapshot_grid(last_date: date, first_year: int = 1990):
    """Annual before 2005, semiannual through 2014, quarterly since 2015."""
    for year in range(first_year, last_date.year + 1):
        months = (
            (1,) if year < 2005 else (1, 7) if year < 2015 else (1, 4, 7, 10)
        )
        for month in months:
            when = date(year, month, 1)
            if when <= last_date:
                yield when


def _three_dates(values):
    if len(values) < 3:
        return ()
    return values[0], values[(len(values) - 1) // 2], values[-1]


def _horizon(when, years):
    year = when.year + years
    return date(
        year, when.month, min(when.day, monthrange(year, when.month)[1])
    )


def _families(corpus):
    """Group known parent/child technologies; reviewers check other aliases."""
    parent = {}

    def root(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    technologies = corpus.view(corpus.latest_date).technologies
    for technology_id, technology in technologies.items():
        for event in technology.relations:
            if event.data.get("relation") != "SUBTECHNOLOGY_OF":
                continue
            target = str(event.data.get("target_id") or "")
            if target in technologies:
                a, b = root(technology_id), root(target)
                parent[max(a, b)] = min(a, b)
    return {key: root(key) for key in parent}


def _review_row(corpus, snapshot, technology_id, families):
    technology = snapshot.technologies[technology_id]
    documents = technology.documents
    recent = sorted(
        documents,
        key=lambda item: (item.first_visible, item.document_id),
        reverse=True,
    )[:5]
    organizations = {
        name
        for item in documents
        for name in (
            *item.version.organizations,
            *item.version.companies,
            *item.version.universities,
        )
    }
    references = [
        {
            "document_id": item.document_id,
            "title": corpus.document_info.get(item.document_id, {}).get(
                "title"
            ),
            "url": corpus.document_info.get(item.document_id, {}).get("url"),
            "date": item.first_visible.isoformat(),
            "source_family": item.family,
            "original_language": item.version.metadata.get("language"),
            "undated": item.undated,
        }
        for item in recent
    ]
    when = snapshot.cutoff
    end_12, end_36 = _horizon(when, 1), _horizon(when, 3)
    return {
        "technology_id": technology_id,
        "technology": technology.label,
        "family_id": families.get(technology_id, technology_id),
        "snapshot_date": when.isoformat(),
        "horizon_12m_end": end_12.isoformat(),
        "horizon_end": end_36.isoformat(),
        "calendar_complete_12m": end_12 <= corpus.latest_date,
        "calendar_complete_36m": end_36 <= corpus.latest_date,
        "document_count": len(documents),
        "first_seen_date": technology.first_seen.isoformat()
        if technology.first_seen
        else "",
        "source_families": "; ".join(
            sorted({item.family for item in documents})
        ),
        "organizations": "; ".join(sorted(organizations)[:20]),
        "recent_document_ids": "; ".join(item.document_id for item in recent),
        "recent_documents": json.dumps(references, ensure_ascii=False),
        "reviewer_state": "",
        "reviewer_signal_12m": "",
        "reviewer_trend_12m": "",
        "reviewer_signal_36m": "",
        "reviewer_trend_36m": "",
        "evidence_notes": "",
    }


def build_pilot_queue(
    corpus: TemporalCorpus,
    technology_limit: int = 100,
    horizon_years: int = 1,
):
    if technology_limit < 1 or horizon_years < 1:
        raise ValueError("technology_limit and horizon_years must be positive")
    if corpus.latest_date is None:
        return []
    last_year = corpus.latest_date.year - horizon_years
    last_month_day = (corpus.latest_date.month, corpus.latest_date.day)
    dates = [
        when
        for when in snapshot_grid(corpus.latest_date)
        if (when.year, when.month, when.day) <= (last_year, *last_month_day)
    ]
    views = [(when, corpus.view(when)) for when in dates]
    available = defaultdict(list)
    for when, snapshot in views:
        for technology_id, technology in snapshot.technologies.items():
            if len(technology.dated_documents) >= 2:
                available[technology_id].append(when)
    latest = corpus.view(corpus.latest_date)
    families = _families(corpus)
    candidates = [
        (technology_id, _three_dates(when))
        for technology_id, when in available.items()
        if len(when) >= 3
    ]
    # A deterministic spread across eras, rather than the 100 biggest hubs.
    eras = defaultdict(list)
    for technology_id, chosen in candidates:
        first = latest.technologies[technology_id].first_seen
        era = (first.year // 10) * 10 if first else 0
        eras[era].append((technology_id, chosen))
    for values in eras.values():
        values.sort(key=lambda item: item[0])
    selected = []
    while len(selected) < technology_limit and any(eras.values()):
        for era in sorted(eras):
            if eras[era] and len(selected) < technology_limit:
                selected.append(eras[era].pop(0))
    view_by_date = dict(views)
    rows = []
    for technology_id, chosen in selected:
        for when in chosen:
            rows.append(
                _review_row(
                    corpus, view_by_date[when], technology_id, families
                )
            )
    return sorted(
        rows, key=lambda row: (row["technology_id"], row["snapshot_date"])
    )


def write_pilot_queue(path: Path, rows, purpose="independent pilot"):
    """Write independent forms for the two reviewers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    for suffix in ("reviewer_1", "reviewer_2"):
        target = path.with_name(f"{path.stem}.{suffix}{path.suffix}")
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=PILOT_FIELDS)
            writer.writeheader()
            writer.writerows(values)
    manifest = {
        "purpose": purpose + ", not machine-generated labels",
        "rows_per_reviewer": len(values),
        "technologies": len({row["technology_id"] for row in values}),
        "allowed_states": [
            "weak",
            "trend",
            "mature",
            "faded",
            "insufficient",
            "rejected",
        ],
        "labels": (
            "Separate 12m and 36m outcomes need future evidence; "
            "leave uncertain cases empty"
        ),
        "family_warning": (
            "Parent/child links grouped; semantic near-duplicates "
            "require manual family review."
        ),
    }
    path.with_name(f"{path.stem}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(values)


def export_pilot_features(
    corpus: TemporalCorpus,
    review_csv: Path,
    output_csv: Path,
    subgraphs_jsonl: Path,
    include_novelty: bool = True,
):
    """Export only reviewed technology/date pairs, without outcome labels."""
    from ...core.config import load_catalog
    from ...graph.subgraphs import sample_neighborhood, write_subgraph_rows
    from ...graph.training import (
        IDENTITY_FIELDS,
        build_snapshot_rows,
        dataset_feature_fields,
    )
    from .labels import TARGETS, _checked_split

    with Path(review_csv).open(encoding="utf-8-sig", newline="") as stream:
        pilot = list(csv.DictReader(stream))
    keys = {(row["technology_id"], row["snapshot_date"]) for row in pilot}
    if len(keys) != len(pilot):
        raise ValueError("Duplicate technology/snapshot in review queue")
    by_date = defaultdict(dict)
    for row in pilot:
        by_date[row["snapshot_date"]][row["technology_id"]] = row
    feature_names = dataset_feature_fields()
    rows, samples = [], []
    for when, requested in sorted(by_date.items()):
        snapshot = corpus.view(date.fromisoformat(when))
        all_features = {
            row["technology_id"]: row
            for row in build_snapshot_rows(
                corpus,
                when,
                min_documents=2,
                include_novelty=include_novelty,
            )
        }
        for technology_id, review in sorted(requested.items()):
            if technology_id not in all_features:
                raise ValueError(
                    f"Pilot technology absent at {when}: {technology_id}"
                )
            row = dict(all_features[technology_id])
            row.update(
                {
                    "horizon_12m_end": review.get("horizon_12m_end", ""),
                    "horizon_end": review["horizon_end"],
                    "signal_12m": "",
                    "trend_12m": "",
                    "signal_36m": "",
                    "trend_36m": "",
                }
            )
            rows.append(row)
            sample = sample_neighborhood(
                snapshot,
                technology_id,
                config=load_catalog("dataset")["neighborhood"],
                features={name: row.get(name) for name in feature_names},
            )
            samples.append(without_unused_embeddings(sample))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        *IDENTITY_FIELDS,
        *feature_names,
        "horizon_12m_end",
        "horizon_end",
        "signal_12m",
        "trend_12m",
        "signal_36m",
        "trend_36m",
    ]
    fields = list(dict.fromkeys(fields))
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_subgraph_rows(subgraphs_jsonl, samples)
    potential = {target: defaultdict(int) for target in TARGETS}
    for row in rows:
        for target, months in TARGETS.items():
            if months == 12 and not row["horizon_12m_end"]:
                continue
            item = dict(row, horizon_36m_end=row["horizon_end"])
            potential[target][_checked_split(item, target)] += 1
    output_csv.with_name(output_csv.name + ".manifest.json").write_text(
        json.dumps(
            {
                "rows": len(rows),
                "features": feature_names,
                "source_data_end": corpus.latest_date.isoformat(),
                "as_known": corpus.as_known,
                "availability_warning": (
                    "Retrospective publication-date view may include "
                    "documents retrieved after the snapshot; use as-known "
                    "exports for a real-time backtest."
                    if not corpus.as_known
                    else None
                ),
                "labels": "empty until independent expert review",
                "potential_rows_by_split": {
                    target: dict(sorted(counts.items()))
                    for target, counts in potential.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(rows)


def export_full_history(
    corpus: TemporalCorpus,
    output_csv: Path,
    review_csv: Path,
    subgraphs_jsonl: Path,
    include_novelty: bool = True,
):
    """Export every documented technology on the historical model grid."""
    from ...core.config import load_catalog
    from ...graph.subgraphs import sample_neighborhood, write_subgraph_rows
    from ...graph.training import (
        IDENTITY_FIELDS,
        build_snapshot_rows,
        dataset_feature_fields,
    )
    from .labels import cohort_cutoffs

    if corpus.latest_date is None:
        raise ValueError("Temporal graph has no dated observations")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_names = dataset_feature_fields()
    fields = list(
        dict.fromkeys(
            (
                *IDENTITY_FIELDS,
                *feature_names,
                "horizon_12m_end",
                "horizon_end",
                "calendar_complete_12m",
                "calendar_complete_36m",
                "signal_12m",
                "trend_12m",
                "signal_36m",
                "trend_36m",
            )
        )
    )
    snapshots = sorted(
        {
            corpus.earliest_date,
            corpus.latest_date,
            *snapshot_grid(corpus.latest_date, corpus.earliest_date.year),
        }
    )
    config = load_catalog("dataset")["neighborhood"]
    families = _families(corpus)
    reviews, metadata, observed = [], [], set()

    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def samples():
            for when in snapshots:
                view = corpus.view(when)
                for row in build_snapshot_rows(
                    corpus,
                    when,
                    min_documents=1,
                    include_novelty=include_novelty,
                ):
                    technology_id = row["technology_id"]
                    review = _review_row(corpus, view, technology_id, families)
                    reviews.append(review)
                    observed.add(technology_id)
                    row.update(
                        {
                            key: review[key]
                            for key in (
                                "horizon_12m_end",
                                "horizon_end",
                                "calendar_complete_12m",
                                "calendar_complete_36m",
                            )
                        }
                    )
                    row.update(
                        {
                            "signal_12m": "",
                            "trend_12m": "",
                            "signal_36m": "",
                            "trend_36m": "",
                        }
                    )
                    writer.writerow(row)
                    metadata.append(
                        {
                            "family_id": review["family_id"],
                            "snapshot_date": review["snapshot_date"],
                            "first_seen_date": review["first_seen_date"],
                            "calendar_complete_12m": review[
                                "calendar_complete_12m"
                            ],
                            "calendar_complete_36m": review[
                                "calendar_complete_36m"
                            ],
                        }
                    )
                    yield without_unused_embeddings(
                        sample_neighborhood(
                            view,
                            technology_id,
                            config=config,
                            features={
                                name: row.get(name) for name in feature_names
                            },
                        )
                    )
                # The next date needs its view, not this bulky typed subgraph.
                view.__dict__.pop("_typed_subgraph_cache", None)
                view.__dict__.pop("_typed_subgraph_adjacency_cache", None)
                view.__dict__.pop("_typed_subgraph_outgoing_cache", None)

        graph_count = write_subgraph_rows(subgraphs_jsonl, samples())

    write_pilot_queue(
        review_csv, reviews, purpose="complete historical review"
    )
    inventory_path = output_csv.with_name(output_csv.stem + ".inventory.csv")
    with inventory_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("technology_id", "technology", "has_snapshot")
        )
        writer.writeheader()
        for technology_id, label in sorted(corpus.labels.items()):
            writer.writerow(
                {
                    "technology_id": technology_id,
                    "technology": label,
                    "has_snapshot": technology_id in observed,
                }
            )
    splits = {}
    for months in (12, 36):
        eligible = [
            row for row in metadata if row[f"calendar_complete_{months}m"]
        ]
        cutoffs = cohort_cutoffs(eligible)
        years = {}
        for row in eligible:
            family = row["family_id"]
            year = int((row["first_seen_date"] or row["snapshot_date"])[:4])
            years[family] = min(year, years.get(family, year))
        counts = defaultdict(int)
        for row in eligible:
            year = years[row["family_id"]]
            part = (
                (
                    "train"
                    if year <= cutoffs[0]
                    else "valid"
                    if year <= cutoffs[1]
                    else "test"
                )
                if cutoffs
                else "unavailable"
            )
            counts[part] += 1
        splits[f"signal_{months}m"] = {
            "calendar_complete_rows": len(eligible),
            "cohort_cutoff_years": cutoffs,
            "potential_rows_by_cohort": dict(counts),
        }
    manifest = {
        "rows": len(metadata),
        "subgraphs": graph_count,
        "technologies_with_snapshots": len(observed),
        "technology_nodes": len(corpus.labels),
        "technologies_without_snapshot": sorted(set(corpus.labels) - observed),
        "snapshots": len(snapshots),
        "first_snapshot": snapshots[0].isoformat(),
        "last_snapshot": snapshots[-1].isoformat(),
        "source_data_end": corpus.latest_date.isoformat(),
        "as_known": corpus.as_known,
        "minimum_documents": 1,
        "labels": "empty until independent expert review",
        "calendar_completion_is_not_source_coverage": True,
        "horizons": splits,
        "review_files": [
            f"{Path(review_csv).stem}.reviewer_{number}{Path(review_csv).suffix}"
            for number in (1, 2)
        ],
        "subgraphs_file": Path(subgraphs_jsonl).name,
        "inventory_file": inventory_path.name,
    }
    output_csv.with_name(output_csv.name + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def without_unused_embeddings(sample):
    """Raw concept vectors are huge and not inputs to this HGT model."""
    for node in sample["nodes"]:
        for key in tuple(node["features"]):
            if key.startswith("embedding_"):
                node["features"].pop(key)
                node["missing_mask"].pop(key, None)
    return sample


def compact_subgraph_file(source: Path, target: Path):
    """Rebuild an existing JSONL/manifest without unused raw embeddings."""
    from ...graph.subgraphs import write_subgraph_rows

    def samples():
        with Path(source).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield without_unused_embeddings(json.loads(line))

    return write_subgraph_rows(target, samples())
