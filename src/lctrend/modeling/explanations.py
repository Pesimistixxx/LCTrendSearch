"""Persist versioned, source-identifiable local model explanations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .catboost_model import explain_catboost
from .dataset import _read_csv
from .hgt_model import build_model, explain_hgt


def _choose(values, technology_id, snapshot_date, date_key):
    matches = [
        row
        for row in values
        if row["technology_id"] == technology_id
        and row[date_key] == snapshot_date
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one technology/snapshot, got {len(matches)}"
        )
    return matches[0]


def _save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def explain_catboost_file(
    model_dir,
    dataset,
    technology_id,
    snapshot_date,
    output,
    target="signal_36m",
    strategy="temporal",
    fold=0,
):
    from catboost import CatBoostClassifier

    model_dir = Path(model_dir)
    stem = (
        "catboost_signal_12m" if target == "signal_12m" else "catboost_signal"
    )
    if strategy == "family":
        stem += f"_family_fold{fold}"
    elif strategy == "cohort":
        stem += "_cohort"
    report = json.loads(
        (model_dir / f"{stem}.json").read_text(encoding="utf-8")
    )
    model = CatBoostClassifier()
    model.load_model(str(model_dir / f"{stem}.cbm"))
    row = _choose(
        _read_csv(dataset), technology_id, snapshot_date, "snapshot_date"
    )
    explanation = explain_catboost(
        model, row, report, top_k=len(report["features"])
    )
    return _save(
        output,
        {
            "technology_id": technology_id,
            "technology": row.get("technology"),
            "snapshot_date": snapshot_date,
            "model": f"{stem}.cbm",
            "target": target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **explanation,
        },
    )


def explain_hgt_file(
    model_dir,
    subgraphs,
    technology_id,
    snapshot_date,
    output,
    target="signal_36m",
    strategy="temporal",
    fold=0,
):
    import torch

    model_dir = Path(model_dir)
    stem = "hgt_signal_12m" if target == "signal_12m" else "hgt_signal"
    if strategy == "family":
        stem += f"_family_fold{fold}"
    elif strategy == "cohort":
        stem += "_cohort"
    report = json.loads(
        (model_dir / f"{stem}.json").read_text(encoding="utf-8")
    )
    model = build_model(len(report["features"]))
    model.load_state_dict(
        torch.load(
            model_dir / f"{stem}.pt", map_location="cpu", weights_only=True
        )
    )
    with Path(subgraphs).open(encoding="utf-8") as stream:
        sample = _choose(
            (json.loads(line) for line in stream if line.strip()),
            technology_id,
            snapshot_date,
            "snapshot",
        )
    explanation = explain_hgt(model, sample, report)
    return _save(
        output,
        {
            "technology_id": technology_id,
            "snapshot_date": snapshot_date,
            "model": f"{stem}.pt",
            "target": target,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **explanation,
        },
    )
