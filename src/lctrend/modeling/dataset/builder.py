"""Level 1 in one call: from the history export to a training sample.

``build_dataset(config)`` reads the history CSV and the subgraphs once and
writes into ``<run>/dataset/``:

- ``training.csv``: labelled snapshots with neighbour aggregates, family,
  split and weight factor;
- ``scoring.csv``: the latest snapshot of every technology, enriched the
  same way, for the final marking;
- ``dataset.json``: what was built and from what (inputs with SHA-256,
  label rule, split dates, selected and dropped features, class counts).

The steps are separate functions, so a notebook can run and inspect them
one by one.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from ..storage import RunLayout, file_digest
from .labels import FEATURES, TRAINING_TARGETS
from .neighbors import (
    NEIGHBOR_FEATURES,
    iter_samples,
    neighbor_features,
    neighbor_groups,
)
from .outcomes import add_months, label_rows

PARTS = ("train", "valid", "test")


def read_history(path: Path) -> List[Dict[str, Any]]:
    csv.field_size_limit(1 << 30)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def enrich_and_relate(
    rows: List[Dict[str, Any]], samples: Iterable[Mapping[str, Any]]
) -> List[tuple]:
    """Add neighbour aggregates in place; return (parent, child) pairs.

    Typed technology links of every sample (parent, child) are collected,
    so families need no second pass over the subgraphs.
    """
    index = {(row["technology_id"], row["snapshot_date"]): row for row in rows}
    related = set()
    for sample in samples:
        row = index.get(
            (str(sample["technology_id"]), str(sample["snapshot"]))
        )
        if row is None:
            continue
        row.update(
            {
                name: "" if value is None else value
                for name, value in neighbor_features(sample, index).items()
            }
        )
        for other in neighbor_groups(sample)["related"]:
            related.add(tuple(sorted((row["technology_id"], other))))
    for row in rows:
        for name in NEIGHBOR_FEATURES:
            row.setdefault(name, "")
    return sorted(related)


def plan_pairs(log: Optional[Mapping[str, Any]]) -> List[tuple]:
    """Duplicates, held versions and review pairs of a deduplication log.

    Versions such as GPT3-13B and GPT3-175B are not merged, but they are
    one family: one of them in train and the other in test would leak.
    Pairs merged in the graph no longer exist as two technologies.
    """
    pairs = []
    for family in (log or {}).get("families", ()):
        for group in family["groups"]:
            pairs += [
                (group["canonical"]["concept_id"], member["concept_id"])
                for member in group["members"]
            ]
        for key in ("held", "review"):
            pairs += [
                (item["canonical"]["concept_id"], item["concept_id"])
                for item in family[key]
            ]
    return pairs


def families(
    technology_ids: Iterable[str], pairs: Iterable[tuple]
) -> Dict[str, str]:
    """Union-find: family id is the smallest technology id in the group."""
    parent = {key: key for key in technology_ids}

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in pairs:
        if left in parent and right in parent:
            a, b = root(left), root(right)
            parent[max(a, b)] = min(a, b)
    return {key: root(key) for key in parent}


def assign_splits(
    rows: List[Dict[str, Any]],
    target: str,
    fractions: Sequence[float],
    family_first_seen: Mapping[str, str],
    embargo: str = "none",
) -> Dict[str, Any]:
    """Split by the date a family first appeared.

    Cut dates put about ``fractions`` of the positive families into each
    part, so valid and test hold only technologies the model has never
    seen, and later than those of train.

    ``embargo="strict"`` also purges every label whose horizon reaches into
    the next part (a point-in-time test). It needs the parts to lie at
    least one horizon apart and discards the late history of old
    technologies; on a small graph it can leave valid empty. ``"none"``
    keeps those rows: it measures transfer to new technologies, not a
    strict forecast.
    """
    if embargo not in ("none", "strict"):
        raise ValueError("embargo must be 'none' or 'strict'")
    months = TRAINING_TARGETS[target]
    positive = sorted(
        {
            family_first_seen[row["family_id"]]
            for row in rows
            if row.get(target) == "1"
        }
    )
    if len(positive) < 3:
        raise ValueError(
            f"{target}: {len(positive)} positive families, need at least 3"
        )

    def cut(share):
        return date.fromisoformat(
            positive[min(len(positive) - 1, round(share * len(positive)))]
        )

    test_start = cut(fractions[0] + fractions[1])
    valid_start = cut(fractions[0])
    if embargo == "strict":
        # A valid row needs T + horizon <= test_start and T >= first seen.
        valid_start = min(valid_start, add_months(test_start, -2 * months))
    for row in rows:
        first = date.fromisoformat(family_first_seen[row["family_id"]])
        part = (
            "train"
            if first < valid_start
            else "valid"
            if first < test_start
            else "test"
        )
        end = date.fromisoformat(row[f"horizon_{months}m_end"])
        if embargo == "strict" and (
            (part == "train" and end > valid_start)
            or (part == "valid" and end > test_start)
        ):
            part = "purged"
        row["split"] = part
    return {
        "valid_start": valid_start.isoformat(),
        "test_start": test_start.isoformat(),
        "embargo": embargo,
        "horizon_months": months,
    }


def _values(rows, name):
    result = []
    for row in rows:
        raw = row.get(name)
        if raw in (None, ""):
            result.append(np.nan)
        elif str(raw).lower() in ("true", "false"):
            result.append(float(str(raw).lower() == "true"))
        else:
            try:
                result.append(float(raw))
            except ValueError:
                result.append(np.nan)
    return np.asarray(result, dtype=float)


def select_features(
    train_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    max_missing_share: float,
    max_abs_correlation: float,
) -> Dict[str, Any]:
    """Drop, on train rows only, empty, constant and duplicate features."""
    selected, dropped, kept_values = [], {}, {}
    for name in candidates:
        values = _values(train_rows, name)
        observed = values[np.isfinite(values)]
        if not len(values) or 1 - len(observed) / len(values) > (
            max_missing_share
        ):
            dropped[name] = "missing"
            continue
        if len(np.unique(observed)) < 2:
            dropped[name] = "constant"
            continue
        twin = None
        for other in selected:
            both = np.isfinite(values) & np.isfinite(kept_values[other])
            if both.sum() < 3:
                continue
            a, b = values[both], kept_values[other][both]
            if a.std() == 0 or b.std() == 0:
                continue
            if abs(np.corrcoef(a, b)[0, 1]) > max_abs_correlation:
                twin = other
                break
        if twin:
            dropped[name] = f"correlated with {twin}"
            continue
        selected.append(name)
        kept_values[name] = values
    return {"selected": selected, "dropped": dropped}


def latest_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        current = latest.get(row["technology_id"])
        if current is None or row["snapshot_date"] > current["snapshot_date"]:
            latest[row["technology_id"]] = row
    return [dict(latest[key]) for key in sorted(latest)]


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def class_counts(rows, target):
    counts = defaultdict(Counter)
    family_sets = defaultdict(lambda: defaultdict(set))
    for row in rows:
        label = row.get(target)
        if label in ("0", "1"):
            counts[row["split"]][label] += 1
            family_sets[row["split"]][label].add(row["family_id"])
    return {
        part: {
            "rows": dict(counts[part]),
            "families": {
                label: len(members)
                for label, members in family_sets[part].items()
            },
        }
        for part in (*PARTS, "purged")
    }


def build_dataset(
    config: Mapping[str, Any], output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    target = config["target"]
    inputs = config["inputs"]
    output_dir = Path(output_dir or RunLayout.at(config.get("run")).dataset)
    history = read_history(inputs["history"])
    related = enrich_and_relate(history, iter_samples(inputs["subgraphs"]))
    plan_path = inputs.get("duplicates_plan")
    plan = (
        json.loads(Path(plan_path).read_text(encoding="utf-8"))
        if plan_path and Path(plan_path).exists()
        else None
    )
    family_of = families(
        {row["technology_id"] for row in history},
        [*related, *plan_pairs(plan)],
    )
    first_seen: Dict[str, str] = {}
    for row in history:
        family = family_of[row["technology_id"]]
        seen = row.get("first_seen_date") or row["snapshot_date"]
        first_seen[family] = min(seen, first_seen.get(family, seen))
    data_end = date.fromisoformat(
        config["labels"].get("data_end")
        or max(row["snapshot_date"] for row in history)
    )
    rule = {
        key: value
        for key, value in config["labels"].items()
        if key != "data_end"
    }
    if config["labels"].get("source", "auto") == "llm":
        from .llm_outcomes import llm_label_rows

        labelled = llm_label_rows(
            history, read_history(config["labels"]["llm_labels"]), rule
        )
        outcome_key = "llm_bucket"
    else:
        labelled = label_rows(history, data_end, rule)
        outcome_key = f"outcome_{TRAINING_TARGETS[target]}m"
    for row in labelled:
        row["family_id"] = family_of[row["technology_id"]]
        row["family_first_seen"] = first_seen[row["family_id"]]
    rows = [row for row in labelled if row.get(target) in ("0", "1")]
    split = assign_splits(
        rows,
        target,
        config["split"]["fractions"],
        first_seen,
        config["split"].get("embargo", "none"),
    )
    rows = [row for row in rows if row["split"] in PARTS]
    features = select_features(
        [row for row in rows if row["split"] == "train"],
        [*FEATURES, *NEIGHBOR_FEATURES],
        config["features"]["max_missing_share"],
        config["features"]["max_abs_correlation"],
    )
    training_path = output_dir / "training.csv"
    scoring_path = output_dir / "scoring.csv"
    write_rows(training_path, rows)
    scoring = latest_rows(history)
    for row in scoring:
        row["family_id"] = family_of[row["technology_id"]]
    write_rows(scoring_path, scoring)
    summary = {
        "created_at": date.today().isoformat(),
        "target": target,
        "data_end": data_end.isoformat(),
        "inputs": {
            key: {"path": str(path), "sha256": file_digest(Path(path))}
            for key, path in inputs.items()
            if path and Path(path).exists()
        },
        "label_rule": rule,
        "history_rows": len(history),
        "active_snapshots": len(labelled),
        "outcomes": dict(Counter(row[outcome_key] for row in labelled)),
        "families": len(set(family_of.values())),
        "family_links": {
            "parent_child": len(related),
            "duplicates_plan": len(plan_pairs(plan)),
        },
        "split": split,
        "classes": class_counts(
            [row for row in labelled if row.get("split")], target
        ),
        "features": features["selected"],
        "dropped_features": features["dropped"],
        "files": {
            "training": str(training_path),
            "scoring": str(scoring_path),
        },
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
