"""Join two expert reviews to historical predictors without target leakage."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# Curated before looking at labels. Dates, current citation counts, and
# mutable repository counters are deliberately not used as predictors.
FEATURES = (
    "technology_age_days",
    "document_count_snapshot_pct",
    "documents_last_year_snapshot_pct",
    "mention_growth_3m",
    "mention_growth_12m",
    "mention_growth_24m",
    "growth_acceleration",
    "burst_score",
    "growth_persistence",
    "active_month_ratio",
    "decline_rate",
    "time_since_last_signal_days",
    "source_type_diversity",
    "independence_group_diversity",
    "source_entropy",
    "independence_group_entropy",
    "recent_source_convergence",
    "evidence_chain_length",
    "country_count",
    "company_count_snapshot_pct",
    "university_count",
    "domain_count",
    "unique_author_count_snapshot_pct",
    "new_author_rate",
    "returning_author_rate",
    "organization_entropy",
    "organization_hhi",
    "top_organization_share",
    "new_company_rate",
    "developer_count",
    "user_count",
    "task_count",
    "commercial_user_count",
    "reliability_weighted_mentions",
    "accepted_mention_ratio",
    "supported_claim_ratio",
    "independent_evidence_family_count",
    "speculative_claim_ratio",
    "duplicate_content_ratio",
    "claim_to_evidence_gap",
    "attention_evidence_gap",
    "independent_refutation_share",
    "economic_evidence_count",
    "investment_evidence_count",
    "grant_count_snapshot_pct",
    "grant_amount_usd_real_snapshot_pct",
    "job_posting_count_snapshot_pct",
    "salary_median_usd_real_snapshot_pct",
    "max_maturity_rank",
    "max_maturity_rank_missing",
    "maturity_growth",
    "implementation_count",
    "patent_family_size",
    "standardization_presence",
    "market_concentration",
    "fulltext_coverage",
    "metadata_only_ratio",
    "provisional_resolution_ratio",
    "repository_source_covered",
    "patent_source_covered",
    "source_coverage_count",
    "new_relation_count",
    "semantic_novelty",
    "nearest_known_distance",
    "branch_new_share",
    "degree_delta_12m",
    "pagerank_delta_12m",
    "community_crossing_count",
    "neighbor_domain_entropy",
    "structural_hole_score",
    "bridge_score",
    "recombination_surprise",
    "data_completeness_score",
)

SPLIT_WINDOWS = {
    "train": (date.min, date(2013, 12, 31)),
    "valid": (date(2016, 1, 1), date(2018, 12, 31)),
    "test": (date(2021, 1, 1), date(2023, 6, 30)),
}
SPLIT_WINDOWS_12M = {
    "train": (date.min, date(2020, 12, 31)),
    "valid": (date(2022, 1, 1), date(2022, 12, 31)),
    "test": (date(2024, 1, 1), date(2025, 6, 30)),
}
TARGETS = {"signal_12m": 12, "signal_36m": 36}


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _label(value):
    if value is None:
        return None
    if str(value).strip() in ("0", "1"):
        return int(value)
    if not str(value).strip():
        return None
    raise ValueError(f"Expected 0, 1 or empty label, got {value!r}")


def _key(row):
    return str(row["technology_id"]), str(row["snapshot_date"])


def _split(when, target="signal_36m"):
    windows = SPLIT_WINDOWS_12M if target == "signal_12m" else SPLIT_WINDOWS
    for name, (start, end) in windows.items():
        if start <= when <= end:
            return name
    return "purged"


def _checked_split(row, target):
    part = _split(date.fromisoformat(row["snapshot_date"]), target)
    windows = SPLIT_WINDOWS_12M if target == "signal_12m" else SPLIT_WINDOWS
    next_start = {
        "train": windows["valid"][0],
        "valid": windows["test"][0],
    }.get(part)
    if (
        next_start
        and date.fromisoformat(row[f"horizon_{TARGETS[target]}m_end"])
        > next_start
    ):
        return "purged_horizon"
    return part


def split_for(row, target="signal_36m", strategy="temporal", fold=0):
    if strategy == "family":
        if not 0 <= fold < 5:
            raise ValueError("Family fold must be 0, 1, 2, 3, or 4")
        if row.get("family_fold") is None:
            raise ValueError("Prepared row lacks family_fold")
        return "valid" if int(row["family_fold"]) == fold else "train"
    if strategy == "cohort":
        key = f"cohort_split_{TARGETS[target]}m"
        if not row.get(key):
            raise ValueError(f"Prepared row lacks {key}")
        return row[key]
    if strategy != "temporal":
        raise ValueError(f"Unknown split strategy: {strategy}")
    months = TARGETS[target]
    return row.get(
        f"split_{months}m", row.get("split") if months == 36 else None
    )


def cohort_cutoffs(rows):
    """Choose year cutoffs near 80/10/10 without splitting families."""
    families = {}
    counts = Counter()
    for row in rows:
        family = row["family_id"]
        year = int((row.get("first_seen_date") or row["snapshot_date"])[:4])
        families[family] = min(year, families.get(family, year))
        counts[family] += 1
    by_year = Counter()
    for family, year in families.items():
        by_year[year] += counts[family]
    years = sorted(by_year)
    if len(years) < 3:
        return None
    cumulative = []
    for year in years:
        cumulative.append(
            (cumulative[-1] if cumulative else 0) + by_year[year]
        )
    total = cumulative[-1]
    _, first, second = min(
        (
            abs(cumulative[i] - 0.8 * total)
            + abs(cumulative[j] - 0.9 * total),
            years[i],
            years[j],
        )
        for i in range(len(years) - 2)
        for j in range(i + 1, len(years) - 1)
    )
    return first, second


def prepare_labeled_rows(
    dataset_rows,
    first_review,
    second_review,
    adjudications=(),
    data_end=None,
):
    """Return audited rows; incomplete/disputed labels remain unlabeled.

    A negative is never inferred from absent future edges. Reviewers must
    explicitly label it and cite evidence in their notes. Disagreements need
    an adjudication row with a recorded decision and rationale.
    """
    if data_end is None:
        raise ValueError("data_end (last verified source date) is required")
    data_end = date.fromisoformat(str(data_end))
    indexed = [list(rows) for rows in (first_review, second_review)]
    if any(len(rows) != len({_key(row) for row in rows}) for rows in indexed):
        raise ValueError("Duplicate technology/snapshot in reviewer file")
    reviews = [{_key(row): row for row in rows} for rows in indexed]
    adjudications = list(adjudications)
    decisions = {_key(row): row for row in adjudications}
    if len(decisions) != len(adjudications):
        raise ValueError("Duplicate adjudication")
    output = []
    for source in dataset_rows:
        row = dict(source)
        key = _key(row)
        when = date.fromisoformat(row["snapshot_date"])
        for months in (12, 36):
            horizon = row.get(f"horizon_{months}m_end")
            if months == 36:
                horizon = horizon or row.get("horizon_end")
            if (
                horizon
                and horizon
                != date(
                    when.year + months // 12, when.month, when.day
                ).isoformat()
            ):
                raise ValueError(f"Incorrect {months}-month horizon for {key}")
        pair = [records.get(key) for records in reviews]
        if not all(pair):
            continue
        if any(
            item.get("family_id") != pair[0].get("family_id") for item in pair
        ):
            raise ValueError(f"Family mismatch for {key}")
        if any(
            item.get("horizon_end") != row.get("horizon_end") for item in pair
        ):
            raise ValueError(f"Horizon mismatch for {key}")
        if row.get("horizon_12m_end") and any(
            item.get("horizon_12m_end") != row["horizon_12m_end"]
            for item in pair
        ):
            raise ValueError(f"12-month horizon mismatch for {key}")
        decision = decisions.get(key)
        if decision:
            if not decision.get("adjudicator") or not decision.get("reason"):
                raise ValueError(
                    f"Adjudication needs person and reason: {key}"
                )
        for months in (12, 36):
            for kind in ("signal", "trend"):
                field = f"{kind}_{months}m"
                values = [
                    _label(item.get(f"reviewer_{field}")) for item in pair
                ]
                result = (
                    _label(decision.get(field))
                    if decision
                    else values[0]
                    if values[0] == values[1]
                    else None
                )
                horizon = row.get(f"horizon_{months}m_end")
                if months == 36:
                    horizon = horizon or row["horizon_end"]
                if not horizon or date.fromisoformat(horizon) > data_end:
                    result = None
                row[field] = result
        if any(
            row[f"signal_{months}m"] is not None for months in (12, 36)
        ) and not all(item.get("evidence_notes", "").strip() for item in pair):
            raise ValueError(f"Labeled row needs both evidence notes: {key}")
        row["family_id"] = pair[0]["family_id"]
        row["family_fold"] = (
            int.from_bytes(
                hashlib.sha256(row["family_id"].encode()).digest()[:8], "big"
            )
            % 5
        )
        row["label_source"] = (
            "expert_adjudicated" if decision else "expert_consensus"
        )
        row["horizon_36m_end"] = row["horizon_end"]
        for target in TARGETS:
            row[f"split_{TARGETS[target]}m"] = (
                _checked_split(row, target)
                if row.get(f"horizon_{TARGETS[target]}m_end")
                else "unavailable"
            )
        row["split"] = row["split_36m"]
        output.append(row)
    # One related family belongs to one part, separately for each horizon.
    for target, months in TARGETS.items():
        first_part = {}
        split_key = f"split_{months}m"
        for row in sorted(output, key=lambda item: item["snapshot_date"]):
            if row[target] is None or row[split_key] not in SPLIT_WINDOWS:
                continue
            family = row["family_id"]
            first_part.setdefault(family, row[split_key])
            if first_part[family] != row[split_key]:
                row[split_key] = "purged_family"
    for row in output:
        row["split"] = row["split_36m"]
    for target, months in TARGETS.items():
        horizon_key = f"horizon_{months}m_end"
        eligible = [
            row
            for row in output
            if row.get(horizon_key)
            and date.fromisoformat(row[horizon_key]) <= data_end
        ]
        cutoffs = cohort_cutoffs(eligible)
        family_years = {}
        for row in eligible:
            family = row["family_id"]
            year = int(
                (row.get("first_seen_date") or row["snapshot_date"])[:4]
            )
            family_years[family] = min(year, family_years.get(family, year))
        eligible_ids = {id(row) for row in eligible}
        for row in output:
            year = family_years.get(row["family_id"])
            row[f"cohort_split_{months}m"] = (
                (
                    "train"
                    if year <= cutoffs[0]
                    else "valid"
                    if year <= cutoffs[1]
                    else "test"
                )
                if cutoffs and id(row) in eligible_ids
                else "unavailable"
            )
    for row in output:
        row["cohort_split"] = row["cohort_split_36m"]
    return output


def prepare_from_files(
    dataset, review_1, review_2, output, data_end, adjudication=None
):
    dataset_rows = _read_csv(dataset)
    rows = prepare_labeled_rows(
        dataset_rows,
        _read_csv(review_1),
        _read_csv(review_2),
        _read_csv(adjudication) if adjudication else [],
        data_end=data_end,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        list(
            dict.fromkeys(
                [
                    *dataset_rows[0].keys(),
                    "horizon_36m_end",
                    "signal_12m",
                    "trend_12m",
                    "family_id",
                    "label_source",
                    "family_fold",
                    "cohort_split",
                    "cohort_split_12m",
                    "cohort_split_36m",
                    "split_12m",
                    "split_36m",
                    "split",
                ]
            )
        )
        if dataset_rows
        else []
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "verified_data_end": str(data_end),
        "labels": dict(Counter(str(row["signal_36m"]) for row in rows)),
        "splits": dict(Counter(row["split"] for row in rows)),
        "labels_12m": dict(Counter(str(row["signal_12m"]) for row in rows)),
        "splits_12m": dict(Counter(row["split_12m"] for row in rows)),
        "family_folds": dict(Counter(str(row["family_fold"]) for row in rows)),
        "cohort_splits": {
            target: dict(
                Counter(row[f"cohort_split_{months}m"] for row in rows)
            )
            for target, months in TARGETS.items()
        },
        "cohort_cutoff_years": {
            target: cohort_cutoffs(
                [
                    row
                    for row in rows
                    if row[f"cohort_split_{months}m"] != "unavailable"
                ]
            )
            for target, months in TARGETS.items()
        },
        "warning": "Both classes are required in train and valid.",
    }
    path.with_name(path.name + ".manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def model_matrix(rows, feature_names=FEATURES):
    """Numerical values only; empty CSV cells become NaN, never zero."""
    import math

    matrix = []
    for row in rows:
        values = []
        for name in feature_names:
            raw = row.get(name)
            if raw in (None, ""):
                values.append(math.nan)
            elif isinstance(raw, bool) or str(raw).lower() in (
                "true",
                "false",
            ):
                values.append(float(str(raw).lower() == "true"))
            else:
                values.append(float(raw))
        matrix.append(values)
    return matrix


def require_trainable(
    rows,
    min_train_families_per_class=20,
    min_valid_families_per_class=10,
    target="signal_36m",
    strategy="temporal",
    fold=0,
):
    if target not in TARGETS:
        raise ValueError(f"Unknown target: {target}")
    months = TARGETS[target]
    windows = SPLIT_WINDOWS_12M if months == 12 else SPLIT_WINDOWS
    for row in rows:
        if _label(row.get(target)) is None:
            continue
        part = split_for(row, target, strategy, fold)
        if part not in ("train", "valid", "test"):
            continue
        if row.get("label_source") not in (
            "expert_consensus",
            "expert_adjudicated",
        ):
            raise ValueError("Training labels require audited expert reviews")
        if strategy == "temporal":
            when = date.fromisoformat(row["snapshot_date"])
            if not windows[part][0] <= when <= windows[part][1]:
                raise ValueError("Snapshot is outside its declared split")
            end = date.fromisoformat(
                row.get(f"horizon_{months}m_end") or row["horizon_end"]
            )
            next_start = {
                "train": windows["valid"][0],
                "valid": windows["test"][0],
            }.get(part)
            if next_start and end > next_start:
                raise ValueError(
                    "Future label horizon overlaps the next split"
                )
    for part in ("train", "valid"):
        groups = defaultdict(set)
        for row in rows:
            label = _label(row.get(target))
            if (
                label is not None
                and split_for(row, target, strategy, fold) == part
            ):
                groups[label].add(row["family_id"])
        counts = {label: len(groups[label]) for label in (0, 1)}
        if not counts[0] or not counts[1]:
            raise ValueError(
                f"{part}: need reviewed positive and negative labels; "
                f"currently 0={counts[0]}, 1={counts[1]}"
            )
        minimum = (
            min_train_families_per_class
            if part == "train"
            else min_valid_families_per_class
        )
        if min(counts.values()) < minimum:
            raise ValueError(
                f"{part}: need at least {minimum} independent families "
                f"per class for a meaningful pilot; currently "
                f"0={counts[0]}, 1={counts[1]}"
            )
    families = defaultdict(set)
    for row in rows:
        if _label(row.get(target)) is None:
            continue
        part = split_for(row, target, strategy, fold)
        if part in ("train", "valid", "test"):
            families[row["family_id"]].add(part)
    overlap = [name for name, parts in families.items() if len(parts) > 1]
    if overlap:
        raise ValueError(
            f"Technology families leak across splits: {overlap[:5]}"
        )
