"""Level 2 in one call: train, compare and apply the models.

``train_pipeline(config)`` reads ``<run>/dataset/`` (built by
``dataset.builder``) and writes ``<run>/models/`` and
``<run>/labeling/scores.csv``:

1. CatBoost on the selected features of the technology and its
   neighbour aggregates: the baseline without a graph model;
2. HGT on the subgraphs (neighbour technologies carry their own rows);
3. stacking, the two combined: CatBoost gets one more feature, the HGT
   probability. On train it is out-of-fold (the fold models never saw the
   row's label); on valid, test and scored rows it is the mean of the fold
   models, none of which saw those labels either. Without this a stacked
   model would learn from HGT outputs fitted to the same labels.

Each model is calibrated on valid (temperature and intercept: class
weights move the prior, the intercept moves it back), its decision
threshold is the F1-optimal one on valid, and test is touched only to
report. The winner is chosen by valid PR-AUC, a rule fixed before any
test number is seen.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..dataset.builder import read_history, write_rows
from ..dataset.labels import (
    AUTO_LABEL_SOURCE,
    EXPERT_LABEL_SOURCES,
    FEATURES,
    LLM_LABEL_SOURCE,
    _read_csv,
    require_trainable,
)
from ..dataset.neighbors import iter_samples
from ..storage import RunLayout
from .catboost_model import (
    _metrics,
    _sigmoid,
    _weights,
    best_f1_threshold,
    calibrated,
    catboost_logits,
    fit_catboost,
    fit_platt,
    train_baselines,
)

logger = logging.getLogger(__name__)

PARTS = ("train", "valid", "test")
STACK_FEATURE = "hgt_oof_probability"


def _key(row):
    return row["technology_id"], row["snapshot_date"]


def family_bootstrap(labels, probabilities, families, draws=500, seed=13):
    """95% interval of PR-AUC resampling whole families, not rows."""
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    groups: Dict[str, List[int]] = {}
    for index, family in enumerate(families):
        groups.setdefault(family, []).append(index)
    names = sorted(groups)
    generator = np.random.default_rng(seed)
    scores = []
    for _ in range(draws):
        chosen = generator.choice(len(names), len(names), replace=True)
        indices = [i for pick in chosen for i in groups[names[pick]]]
        if len(set(labels[indices])) == 2:
            scores.append(
                average_precision_score(
                    labels[indices], probabilities[indices]
                )
            )
    if not scores:
        return None
    return [
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    ]


def evaluate(parts, logits, target):
    """Temperature and threshold on valid; metrics of valid and test."""
    labels = {
        part: [int(row[target]) for row in parts[part]] for part in PARTS
    }
    temperature, intercept = fit_platt(logits["valid"], labels["valid"])
    probabilities = {
        part: calibrated(logits[part], temperature, intercept)
        for part in logits
    }
    threshold = best_f1_threshold(labels["valid"], probabilities["valid"])
    metrics = {}
    for part in ("valid", "test"):
        if not parts[part]:
            continue
        metrics[part] = _metrics(
            labels[part],
            probabilities[part],
            np.ones(len(labels[part])),
            threshold,
        )
        metrics[part]["pr_auc_family_ci95"] = family_bootstrap(
            labels[part],
            probabilities[part],
            [row["family_id"] for row in parts[part]],
        )
    return {
        "temperature_valid_only": temperature,
        "intercept_valid_only": intercept,
        "threshold_valid_f1": float(threshold),
        "metrics": metrics,
    }, probabilities


def _fold(family, folds):
    digest = hashlib.sha256(str(family).encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _hgt_items(rows, graphs, target):
    weights = _weights(rows, target) if rows else []
    return [
        (graphs[_key(row)], int(row[target]), weight, None)
        for row, weight in zip(rows, weights)
    ]


def train_pipeline(
    config: Mapping[str, Any],
    dataset_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    scores_path: Optional[Path] = None,
) -> Dict[str, Any]:
    layout = RunLayout.at(config.get("run"))
    dataset_dir = Path(dataset_dir or layout.dataset)
    models_dir = Path(output_dir or layout.models)
    models_dir.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(
        (dataset_dir / "dataset.json").read_text(encoding="utf-8")
    )
    target = dataset["target"]
    features = list(dataset["features"])
    rows = _read_csv(dataset_dir / "training.csv")
    scoring = _read_csv(dataset_dir / "scoring.csv")
    minimum = config["minimum_families_per_class"]
    require_trainable(
        rows,
        minimum["train"],
        minimum["valid"],
        target,
        "prepared",
        label_sources=(
            AUTO_LABEL_SOURCE,
            LLM_LABEL_SOURCE,
            *EXPERT_LABEL_SOURCES,
        ),
        embargo=dataset["split"].get("embargo") == "strict",
    )
    parts = {
        part: [row for row in rows if row["split"] == part] for part in PARTS
    }
    report: Dict[str, Any] = {
        "target": target,
        "dataset": str(dataset_dir / "dataset.json"),
        "split": dataset["split"],
        "rows": {part: len(values) for part, values in parts.items()},
        "positive_families": {
            part: len(
                {row["family_id"] for row in values if row[target] == "1"}
            )
            for part, values in parts.items()
        },
        "models": {},
    }
    # Model name -> {(technology_id, snapshot_date): probability}.
    scores: Dict[str, Dict[tuple, float]] = {}

    # 1. CatBoost on the feature matrix.
    settings = config["catboost"]
    catboost = fit_catboost(
        parts["train"],
        parts["valid"],
        features,
        target,
        settings["iterations"],
        settings["depth"],
        settings["learning_rate"],
        settings["seed"],
    )
    logits = {
        part: catboost_logits(catboost, parts[part], features)
        for part in PARTS
    }
    logits["scoring"] = catboost_logits(catboost, scoring, features)
    result, probabilities = evaluate(parts, logits, target)
    catboost.save_model(str(models_dir / "catboost.cbm"))
    report["models"]["catboost"] = {
        **result,
        "features": features,
        "best_iteration": catboost.get_best_iteration(),
        "baseline_medians_train_only": train_baselines(
            parts["train"], features
        ),
        "importance": dict(
            sorted(
                zip(features, catboost.get_feature_importance().tolist()),
                key=lambda item: -item[1],
            )[:20]
        ),
    }
    scores["catboost"] = _by_key(scoring, probabilities["scoring"])
    # Model name -> {(technology_id, snapshot_date): valid/test probability}.
    predictions = {"catboost": _part_probabilities(parts, probabilities)}

    if config["hgt"]["enabled"]:
        hgt_report, hgt_scores, hgt_predictions = _train_graph_models(
            config, dataset, parts, scoring, features, target, models_dir
        )
        report["models"].update(hgt_report)
        scores.update(hgt_scores)
        predictions.update(hgt_predictions)

    comparable = {
        name: model["metrics"]["valid"].get("pr_auc", -1)
        for name, model in report["models"].items()
    }
    winner = max(comparable, key=comparable.get)
    report["winner"] = {
        "model": winner,
        "rule": "highest valid PR-AUC, fixed before looking at test",
        "valid_pr_auc": comparable,
    }
    sources = sorted({row.get("label_source") for row in rows})
    report["label_sources"] = sources
    report["warning"] = (
        "Labels are "
        + ", ".join(sources)
        + ": an LLM's judgement or each technology's own future in our "
        "corpus, not expert truth. Few positive families make every "
        "metric wide: read the family bootstrap intervals."
    )
    (models_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        models_dir / "predictions.csv", parts, predictions, target
    )
    _write_scores(
        scores_path or layout.labeling / "scores.csv",
        scoring,
        scores,
        report["models"],
        winner,
    )
    return report


def prepare_graphs(dataset, parts, scoring, features, neighbours=True):
    """HGT inputs of every train/valid/test and scored row.

    Root technologies carry the selected features of the history; with
    ``neighbours`` so do the neighbour technologies of each subgraph (off:
    neighbours enter as blank vectors, the ablation of that fix). Rows
    without a subgraph are dropped from ``parts`` and ``scoring``.
    """
    from .hgt_model import fit_scaler, hgt_data

    history = read_history(dataset["inputs"]["history"]["path"])
    technology_rows = (
        {_key(row): row for row in history} if neighbours else None
    )
    names = [name for name in FEATURES if name in features]
    scaler = fit_scaler(parts["train"], names)
    wanted = {_key(row) for part in PARTS for row in parts[part]}
    wanted |= {_key(row) for row in scoring}
    graphs = {}
    for sample in iter_samples(dataset["inputs"]["subgraphs"]["path"]):
        key = (str(sample["technology_id"]), str(sample["snapshot"]))
        if key in wanted:
            graphs[key] = hgt_data(sample, names, scaler, technology_rows)
    missing = [key for key in wanted if key not in graphs]
    if missing:
        logger.warning("%d rows have no subgraph; dropped", len(missing))
        parts = {
            part: [row for row in values if _key(row) in graphs]
            for part, values in parts.items()
        }
        scoring = [row for row in scoring if _key(row) in graphs]
    return names, scaler, graphs, parts, scoring


def _train_graph_models(
    config, dataset, parts, scoring, features, target, models_dir
):
    """HGT and, if enabled, the stacked CatBoost."""
    import torch

    from .hgt_model import fit_hgt, predict_logits

    settings = config["hgt"]
    names, scaler, graphs, parts, scoring = prepare_graphs(
        dataset, parts, scoring, features
    )

    def graphs_of(rows):
        return [graphs[_key(row)] for row in rows]

    fit = {
        "epochs": settings["epochs"],
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "seed": settings["seed"],
    }
    main = fit_hgt(
        _hgt_items(parts["train"], graphs, target),
        len(names),
        valid=[
            (graphs[_key(row)], int(row[target])) for row in parts["valid"]
        ],
        patience=settings["patience"],
        **fit,
    )
    torch.save(main["state"], models_dir / "hgt.pt")
    logits = {
        part: predict_logits(main["model"], graphs_of(parts[part]))
        for part in PARTS
    }
    logits["scoring"] = predict_logits(main["model"], graphs_of(scoring))
    result, probabilities = evaluate(parts, logits, target)
    report = {
        "hgt": {
            **result,
            "features": names,
            "scaler_train_only": scaler,
            "best_epoch": main["best_epoch"],
            "valid_pr_auc_by_epoch": main["valid_pr_auc_by_epoch"],
        }
    }
    scores = {"hgt": _by_key(scoring, probabilities["scoring"])}
    predictions = {"hgt": _part_probabilities(parts, probabilities)}
    if not config["stacking"]["enabled"]:
        return report, scores, predictions

    folds = config["stacking"]["folds"]
    train = parts["train"]
    out_of_fold = np.zeros(len(train))
    shared = {part: np.zeros(len(parts[part])) for part in ("valid", "test")}
    shared["scoring"] = np.zeros(len(scoring))
    fold_of = [_fold(row["family_id"], folds) for row in train]
    for fold in range(folds):
        inside = [row for row, f in zip(train, fold_of) if f != fold]
        held = [i for i, f in enumerate(fold_of) if f == fold]
        model = fit_hgt(
            _hgt_items(inside, graphs, target),
            len(names),
            **{
                **fit,
                "epochs": main["best_epoch"],
                "seed": fit["seed"] + fold,
            },
        )["model"]
        torch.save(model.state_dict(), models_dir / f"hgt_fold{fold}.pt")
        if held:
            out_of_fold[held] = predict_logits(
                model, graphs_of([train[i] for i in held])
            )
        for part in ("valid", "test"):
            shared[part] += predict_logits(model, graphs_of(parts[part]))
        shared["scoring"] += predict_logits(model, graphs_of(scoring))
    stacked_parts = {
        "train": [
            {**row, STACK_FEATURE: float(_sigmoid(value))}
            for row, value in zip(train, out_of_fold)
        ],
        **{
            part: [
                {**row, STACK_FEATURE: float(_sigmoid(value / folds))}
                for row, value in zip(parts[part], shared[part])
            ]
            for part in ("valid", "test")
        },
    }
    stacked_scoring = [
        {**row, STACK_FEATURE: float(_sigmoid(value / folds))}
        for row, value in zip(scoring, shared["scoring"])
    ]
    stacked_features = [*features, STACK_FEATURE]
    settings = config["catboost"]
    catboost = fit_catboost(
        stacked_parts["train"],
        stacked_parts["valid"],
        stacked_features,
        target,
        settings["iterations"],
        settings["depth"],
        settings["learning_rate"],
        settings["seed"],
    )
    catboost.save_model(str(models_dir / "stacked.cbm"))
    logits = {
        part: catboost_logits(catboost, stacked_parts[part], stacked_features)
        for part in PARTS
    }
    logits["scoring"] = catboost_logits(
        catboost, stacked_scoring, stacked_features
    )
    result, probabilities = evaluate(stacked_parts, logits, target)
    importance = dict(
        zip(stacked_features, catboost.get_feature_importance().tolist())
    )
    report["stacked"] = {
        **result,
        "features": stacked_features,
        "folds": folds,
        "fold_epochs": main["best_epoch"],
        "hgt_feature_importance": importance[STACK_FEATURE],
        "hgt_feature_rank": sorted(
            importance, key=lambda name: -importance[name]
        ).index(STACK_FEATURE)
        + 1,
    }
    scores["stacked"] = _by_key(stacked_scoring, probabilities["scoring"])
    predictions["stacked"] = _part_probabilities(stacked_parts, probabilities)
    return report, scores, predictions


def _part_probabilities(parts, probabilities):
    return {
        **_by_key(parts["valid"], probabilities["valid"]),
        **_by_key(parts["test"], probabilities["test"]),
    }


def _write_predictions(path, parts, predictions, target):
    """Valid and test rows with the calibrated probability of each model:
    the input of PR curves and calibration plots."""
    rows = []
    for part in ("valid", "test"):
        for row in parts[part]:
            key = _key(row)
            record = {
                "technology_id": row["technology_id"],
                "technology": row.get("technology"),
                "snapshot_date": row["snapshot_date"],
                "family_id": row["family_id"],
                "split": part,
                "label": row[target],
            }
            for name, values in predictions.items():
                value = values.get(key)
                record[f"p_{name}"] = "" if value is None else round(value, 6)
            rows.append(record)
    write_rows(Path(path), rows)


def _by_key(rows, values):
    return {_key(row): float(value) for row, value in zip(rows, values)}


def _write_scores(path, scoring, scores, models, winner):
    """Every technology at its latest snapshot, most probable first.

    A technology a graph model could not score (no subgraph) keeps its
    CatBoost probability columns only and no winner probability.
    """
    rows = []
    for row in scoring:
        key = _key(row)
        record = {
            "technology_id": row["technology_id"],
            "technology": row.get("technology"),
            "snapshot_date": row["snapshot_date"],
            "family_id": row.get("family_id"),
        }
        for name, values in scores.items():
            value = values.get(key)
            record[f"p_{name}"] = "" if value is None else round(value, 6)
        chosen = scores[winner].get(key)
        record["model"] = winner
        if chosen is None:
            record.update(probability="", signal="", above_0_75="")
        else:
            record["probability"] = round(chosen, 6)
            record["signal"] = int(
                chosen >= models[winner]["threshold_valid_f1"]
            )
            record["above_0_75"] = int(chosen >= 0.75)
        rows.append(record)
    rows.sort(
        key=lambda item: (
            -(item["probability"] if item["probability"] != "" else -1)
        )
    )
    write_rows(Path(path), rows)
