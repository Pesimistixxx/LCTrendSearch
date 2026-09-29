"""Calibrated CatBoost baseline and honest local feature explanations."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .dataset import (
    FEATURES,
    TARGETS,
    _read_csv,
    model_matrix,
    require_trainable,
    split_for,
)


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    return 1 / (1 + np.exp(-np.clip(values, -50, 50)))


def fit_temperature(logits, labels, weights=None):
    """Weighted log-loss calibration; never fit this on test rows."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights)
    if len(labels) < 2 or len(set(labels)) < 2:
        raise ValueError("Temperature needs both validation classes")
    candidates = np.exp(np.linspace(math.log(0.2), math.log(5.0), 201))
    losses = []
    for value in candidates:
        p = np.clip(_sigmoid(logits / value), 1e-9, 1 - 1e-9)
        losses.append(
            np.average(
                -(labels * np.log(p) + (1 - labels) * np.log1p(-p)),
                weights=weights,
            )
        )
    return float(candidates[int(np.argmin(losses))])


def _weights(rows, target="signal_36m"):
    """One total weight per technology, then balance classes within era."""
    counts = Counter(row["technology_id"] for row in rows)
    base = np.asarray([1 / counts[row["technology_id"]] for row in rows])
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        era = int(row["snapshot_date"][:4]) // 10
        groups[(era, int(row[target]))].append(index)
    result = base.copy()
    for era in {group[0] for group in groups}:
        indices = [
            i
            for (current, _), items in groups.items()
            if current == era
            for i in items
        ]
        total = sum(base[i] for i in indices)
        for label in (0, 1):
            subset = groups.get((era, label), [])
            weight = sum(base[i] for i in subset)
            if weight:
                result[subset] *= total / (2 * weight)
    return result.tolist()


def _selected_features(rows):
    available = []
    for name in FEATURES:
        values = {row[name] for row in rows if row.get(name) not in (None, "")}
        if len(values) > 1:
            available.append(name)
    if len(available) < 2:
        raise ValueError("Too few varying historical predictors")
    return available


def _metrics(labels, probs, weights):
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probs, dtype=float)
    values = {
        "rows": int(len(labels)),
        "positive": int(sum(labels)),
        "accuracy_0_5": float(
            accuracy_score(labels, probabilities >= 0.5, sample_weight=weights)
        ),
        "brier": float(
            brier_score_loss(labels, probabilities, sample_weight=weights)
        ),
    }
    total = float(sum(weights))
    calibration_error = 0.0
    for bucket in range(10):
        lower, upper = bucket / 10, (bucket + 1) / 10
        selected = (probabilities >= lower) & (
            probabilities < upper if bucket < 9 else probabilities <= upper
        )
        selected_weights = np.asarray(weights)[selected]
        amount = float(sum(selected_weights))
        if amount:
            predicted = np.average(
                probabilities[selected], weights=selected_weights
            )
            observed = np.average(labels[selected], weights=selected_weights)
            calibration_error += amount / total * abs(predicted - observed)
    values["ece_10_bins"] = float(calibration_error)
    if len(set(labels)) == 2:
        values["pr_auc"] = float(
            average_precision_score(
                labels, probabilities, sample_weight=weights
            )
        )
        values["roc_auc"] = float(
            roc_auc_score(labels, probabilities, sample_weight=weights)
        )
    return values


def train_catboost(
    rows,
    output_dir,
    iterations=800,
    min_train_families_per_class=20,
    min_valid_families_per_class=10,
    target="signal_36m",
    strategy="temporal",
    fold=0,
):
    """Train only with checked labels and disjoint train/valid families."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError as exc:
        raise ImportError("Install lctrend[models] to train CatBoost") from exc
    require_trainable(
        rows,
        min_train_families_per_class,
        min_valid_families_per_class,
        target,
        strategy,
        fold,
    )
    months = TARGETS[target]
    parts = {
        name: [
            row
            for row in rows
            if split_for(row, target, strategy, fold) == name
            and str(row.get(target)) in ("0", "1")
        ]
        for name in ("train", "valid", "test")
    }
    features = _selected_features(parts["train"])
    matrices = {
        name: model_matrix(values, features) for name, values in parts.items()
    }
    labels = {
        name: [int(row[target]) for row in values]
        for name, values in parts.items()
    }
    weights = {
        name: _weights(values, target) for name, values in parts.items()
    }
    train = Pool(
        matrices["train"],
        labels["train"],
        weight=weights["train"],
        feature_names=features,
    )
    valid = Pool(
        matrices["valid"],
        labels["valid"],
        weight=weights["valid"],
        feature_names=features,
    )
    model = CatBoostClassifier(
        loss_function="Logloss",
        depth=6,
        learning_rate=0.05,
        iterations=iterations,
        random_seed=13,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train, eval_set=valid, early_stopping_rounds=100, use_best_model=True
    )
    valid_logits = model.predict(valid, prediction_type="RawFormulaVal")
    temperature = fit_temperature(valid_logits, labels["valid"])
    results = {}
    for name in ("valid", "test"):
        if not parts[name]:
            continue
        pool = Pool(matrices[name], feature_names=features)
        logits = model.predict(pool, prediction_type="RawFormulaVal")
        results[name] = _metrics(
            labels[name],
            _sigmoid(logits / temperature),
            np.ones(len(labels[name])),
        )
    train_matrix = np.asarray(matrices["train"], dtype=float)
    baselines = {}
    for index, name in enumerate(features):
        finite = train_matrix[:, index][np.isfinite(train_matrix[:, index])]
        baselines[name] = float(np.median(finite)) if len(finite) else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "catboost_signal" if months == 36 else "catboost_signal_12m"
    if strategy == "family":
        stem += f"_family_fold{fold}"
    elif strategy == "cohort":
        stem += "_cohort"
    model.save_model(str(output_dir / f"{stem}.cbm"))
    report = {
        "model": f"CatBoost {target}",
        "target": target,
        "split_strategy": strategy,
        "family_fold": fold if strategy == "family" else None,
        "features": features,
        "baseline_medians_train_only": baselines,
        "temperature_valid_only": temperature,
        "best_iteration": model.get_best_iteration(),
        "metrics": results,
        "warning": (
            "Feature replacements are model comparisons, not causal effects."
        ),
    }
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def train_from_file(
    dataset,
    output_dir,
    iterations=800,
    target="signal_36m",
    strategy="temporal",
    fold=0,
):
    return train_catboost(
        _read_csv(dataset),
        output_dir,
        iterations,
        target=target,
        strategy=strategy,
        fold=fold,
    )


def explain_catboost(model, row, report, top_k=12):
    """SHAP logit decomposition plus calibrated probability comparisons."""
    from catboost import Pool

    names = report["features"]
    values = model_matrix([row], names)[0]
    pool = Pool([values], feature_names=names)
    raw = float(model.predict(pool, prediction_type="RawFormulaVal")[0])
    temperature = report["temperature_valid_only"]
    probability = float(_sigmoid(raw / temperature))
    shap = model.get_feature_importance(pool, type="ShapValues")[0]
    explanations = []
    for index in sorted(
        range(len(names)), key=lambda i: abs(shap[i]), reverse=True
    )[:top_k]:
        name = names[index]
        baseline = report["baseline_medians_train_only"][name]
        if baseline is None:
            continue
        changed = list(values)
        changed[index] = baseline
        counterfactual_raw = float(
            model.predict(
                Pool([changed], feature_names=names),
                prediction_type="RawFormulaVal",
            )[0]
        )
        other_probability = float(_sigmoid(counterfactual_raw / temperature))
        explanations.append(
            {
                "feature": name,
                "value": None if math.isnan(values[index]) else values[index],
                "train_median": baseline,
                "shap_logit": float(shap[index]),
                "probability_point_difference_vs_train_median": round(
                    100 * (probability - other_probability), 3
                ),
            }
        )
    return {
        "probability": probability,
        "base_logit": float(shap[-1]),
        "features": explanations,
        "explanation_method": (
            "SHAP logit and one-feature replacement, not causality"
        ),
    }
