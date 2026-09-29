"""Controlled comparisons: does the graph (HGT) beat the table (CatBoost)?

Every experiment uses the same dataset, split, training code and
evaluation as the pipeline (``pipeline.evaluate``: temperature on valid,
metrics on valid and test, test PR-AUC with a family bootstrap interval),
so differences come from the model, not from the setup:

- ``random``: the no-skill level, PR-AUC equal to the positive rate;
- ``rule``: one feature used as a score, the feature and its direction
  chosen on valid — what a hand-made rule could reach;
- ``catboost_own``: CatBoost on the technology's own features only;
- ``catboost``: CatBoost with neighbour aggregates (the pipeline model);
- ``hgt``: HGT on the subgraphs, neighbours carrying their own features;
- ``hgt_blank_neighbours``: HGT with neighbours as blank vectors — how much
  the neighbours' features matter;
- ``catboost_shuffled``: CatBoost trained on shuffled labels. It must fall
  to the random level; if it does not, something leaks into the features.

Models with randomness run over several seeds, so a difference smaller than
the spread between seeds is not a finding.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..dataset.labels import FEATURES, _read_csv
from ..storage import RunLayout
from .catboost_model import catboost_logits, fit_catboost
from .pipeline import PARTS, _hgt_items, _key, evaluate, prepare_graphs

logger = logging.getLogger(__name__)

RULE_CANDIDATES = (
    "documents_last_year_snapshot_pct",
    "document_count_snapshot_pct",
    "mention_growth_12m",
    "burst_score",
    "independence_group_diversity",
    "new_author_rate",
    "semantic_novelty",
    "technology_age_days",
    "nb_comentioned_count",
    "nb_growing_share",
)


def load(config: Mapping[str, Any]) -> Dict[str, Any]:
    dataset_dir = RunLayout.at(config.get("run")).dataset
    dataset = json.loads(
        (dataset_dir / "dataset.json").read_text(encoding="utf-8")
    )
    rows = _read_csv(dataset_dir / "training.csv")
    return {
        "dataset": dataset,
        "target": dataset["target"],
        "features": list(dataset["features"]),
        "parts": {
            part: [row for row in rows if row["split"] == part]
            for part in PARTS
        },
        "scoring": _read_csv(dataset_dir / "scoring.csv"),
    }


def _summary(name, seed, result) -> Dict[str, Any]:
    metrics = result["metrics"]
    return {
        "experiment": name,
        "seed": seed,
        "valid_pr_auc": metrics["valid"].get("pr_auc"),
        "test_pr_auc": metrics["test"].get("pr_auc"),
        "test_pr_auc_ci95": metrics["test"].get("pr_auc_family_ci95"),
        "test_roc_auc": metrics["test"].get("roc_auc"),
        "test_f1": metrics["test"].get("at_valid_f1_threshold", {}).get("f1"),
        "test_positive_rate": metrics["test"].get("positive_rate"),
    }


def _value(row, name):
    raw = row.get(name)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def rule(study) -> Dict[str, Any]:
    """The best single feature as a score, chosen on valid."""
    from sklearn.metrics import average_precision_score

    parts, target = study["parts"], study["target"]
    labels = [int(row[target]) for row in parts["valid"]]
    best = None
    for name in RULE_CANDIDATES:
        values = np.asarray([_value(row, name) for row in parts["valid"]])
        if np.isnan(values).all():
            continue
        filled = np.nan_to_num(values, nan=np.nanmedian(values))
        for sign in (1, -1):
            score = average_precision_score(labels, sign * filled)
            if best is None or score > best[0]:
                best = (score, name, sign)
    _, name, sign = best
    logits = {}
    for part in PARTS:
        values = np.asarray([_value(row, name) for row in parts[part]])
        filled = np.nan_to_num(values, nan=np.nanmedian(values))
        # Standardise so the temperature fit sees a logit-like scale.
        logits[part] = sign * (filled - filled.mean()) / (filled.std() or 1)
    result, _ = evaluate(parts, logits, target)
    summary = _summary("rule", None, result)
    summary["rule"] = f"{'+' if sign > 0 else '−'}{name}"
    return summary


def catboost_run(study, config, features, seed, shuffle=False):
    parts, target = study["parts"], study["target"]
    train = parts["train"]
    if shuffle:
        labels = [row[target] for row in train]
        random.Random(seed).shuffle(labels)
        train = [{**row, target: label} for row, label in zip(train, labels)]
    settings = config["catboost"]
    model = fit_catboost(
        train,
        parts["valid"],
        features,
        target,
        settings["iterations"],
        settings["depth"],
        settings["learning_rate"],
        seed,
    )
    logits = {
        part: catboost_logits(model, parts[part], features) for part in PARTS
    }
    result, _ = evaluate(parts, logits, target)
    return result


def hgt_run(study, config, graphs_bundle, seed):
    from .hgt_model import fit_hgt, predict_logits

    names, _, graphs, parts, _ = graphs_bundle
    target = study["target"]
    settings = config["hgt"]
    fitted = fit_hgt(
        _hgt_items(parts["train"], graphs, target),
        len(names),
        valid=[
            (graphs[_key(row)], int(row[target])) for row in parts["valid"]
        ],
        epochs=settings["epochs"],
        patience=settings["patience"],
        batch_size=settings["batch_size"],
        learning_rate=settings["learning_rate"],
        seed=seed,
    )
    logits = {
        part: predict_logits(
            fitted["model"], [graphs[_key(row)] for row in parts[part]]
        )
        for part in PARTS
    }
    result, _ = evaluate(parts, logits, target)
    return result, fitted["best_epoch"]


def run_research(
    config: Mapping[str, Any],
    seeds: Sequence[int] = (13, 14, 15, 16, 17),
    hgt_seeds: Sequence[int] = (13, 14, 15),
    output: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Every experiment; results are also written to <run>/research/."""
    study = load(config)
    target, parts = study["target"], study["parts"]
    features = study["features"]
    own = [name for name in features if name in FEATURES]
    results: List[Dict[str, Any]] = []
    test_rate = float(np.mean([int(row[target]) for row in parts["test"]]))
    valid_rate = float(np.mean([int(row[target]) for row in parts["valid"]]))
    results.append(
        {
            "experiment": "random",
            "seed": None,
            "valid_pr_auc": valid_rate,
            "test_pr_auc": test_rate,
            "test_pr_auc_ci95": None,
            "test_roc_auc": 0.5,
            "test_f1": None,
            "test_positive_rate": test_rate,
        }
    )
    results.append(rule(study))
    for seed in seeds:
        logger.info("CatBoost seed %s", seed)
        results.append(
            _summary(
                "catboost_own", seed, catboost_run(study, config, own, seed)
            )
        )
        results.append(
            _summary(
                "catboost", seed, catboost_run(study, config, features, seed)
            )
        )
        results.append(
            _summary(
                "catboost_shuffled",
                seed,
                catboost_run(study, config, features, seed, shuffle=True),
            )
        )
    for neighbours, name in ((True, "hgt"), (False, "hgt_blank_neighbours")):
        bundle = prepare_graphs(
            study["dataset"],
            parts,
            study["scoring"],
            features,
            neighbours=neighbours,
        )
        for seed in hgt_seeds:
            logger.info("%s seed %s", name, seed)
            result, epoch = hgt_run(study, config, bundle, seed)
            summary = _summary(name, seed, result)
            summary["best_epoch"] = epoch
            results.append(summary)
    output = Path(
        output or RunLayout.at(config.get("run")).root / "research.json"
    )
    output.write_text(
        json.dumps(
            {
                "target": target,
                "rows": {part: len(values) for part, values in parts.items()},
                "own_features": len(own),
                "all_features": len(features),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results
