"""Calibrated probabilities of a trained CatBoost for every technology.

By default each technology is scored at its latest snapshot in the
dataset, the state a search would show today. The model and temperature
are read exactly as training saved them; nothing is refitted here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

from ..dataset.labels import _read_csv, model_matrix
from ..training.catboost_model import _sigmoid, model_stem

SCORE_FIELDS = (
    "technology_id",
    "technology",
    "snapshot_date",
    "probability",
    "model",
    "target",
)


def latest_rows(rows, snapshot: Optional[str] = None):
    """One row per technology: at ``snapshot`` or at its latest date."""
    chosen: Dict[str, dict] = {}
    for row in rows:
        if snapshot and row["snapshot_date"] != snapshot:
            continue
        current = chosen.get(row["technology_id"])
        if current is None or row["snapshot_date"] > current["snapshot_date"]:
            chosen[row["technology_id"]] = row
    return [chosen[key] for key in sorted(chosen)]


def score_catboost_file(
    model_dir: Path,
    dataset: Path,
    output: Path,
    target: str = "signal_36m",
    strategy: str = "temporal",
    fold: int = 0,
    snapshot: Optional[str] = None,
) -> Dict:
    from catboost import CatBoostClassifier, Pool

    stem = model_stem("catboost", target, strategy, fold)
    report = json.loads(
        (Path(model_dir) / f"{stem}.json").read_text(encoding="utf-8")
    )
    model = CatBoostClassifier()
    model.load_model(str(Path(model_dir) / f"{stem}.cbm"))
    rows = latest_rows(_read_csv(dataset), snapshot)
    names = report["features"]
    missing = [name for name in names if rows and name not in rows[0]]
    if missing:
        raise ValueError(
            f"Dataset lacks model features: {missing[:5]}; score the same "
            "kind of dataset the model was trained on"
        )
    logits = (
        model.predict(
            Pool(model_matrix(rows, names), feature_names=names),
            prediction_type="RawFormulaVal",
        )
        if rows
        else []
    )
    probabilities = _sigmoid(
        [
            value / report["temperature_valid_only"]
            + report.get("intercept_valid_only", 0.0)
            for value in logits
        ]
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    "technology_id": row["technology_id"],
                    "technology": row.get("technology"),
                    "snapshot_date": row["snapshot_date"],
                    "probability": round(float(probability), 6),
                    "model": f"{stem}.cbm",
                    "target": target,
                }
            )
    return {
        "technologies": len(rows),
        "above_0_75": int(sum(float(p) > 0.75 for p in probabilities)),
        "model": f"{stem}.cbm",
    }
