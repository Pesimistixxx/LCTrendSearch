import csv
import json
from datetime import date

import pytest

from lctrend.modeling.config import load_config
from lctrend.modeling.dataset.builder import (
    assign_splits,
    families,
    plan_pairs,
    select_features,
)
from lctrend.modeling.dataset.outcomes import add_months, label_rows


def _history_row(technology, when, documents, last_year, groups, **extra):
    return {
        "technology_id": technology,
        "snapshot_date": when,
        "first_seen_date": extra.pop("first_seen", "2019-01-01"),
        "document_count": str(documents),
        "documents_last_year": str(last_year),
        "independence_group_diversity": str(groups),
        "documents_last_year_snapshot_pct": str(extra.pop("pct", 0.5)),
        **extra,
    }


def test_add_months_keeps_month_ends_valid():
    assert add_months(date(2020, 1, 31), 1) == date(2020, 2, 29)
    assert add_months(date(2021, 11, 30), 12) == date(2022, 11, 30)
    assert add_months(date(2021, 1, 1), -36) == date(2018, 1, 1)


def test_labels_compare_the_snapshot_with_its_horizon_end():
    history = [
        # Adopted: two new documents from a new group within a year.
        _history_row("a", "2020-01-01", 1, 1, 1),
        _history_row("a", "2021-01-01", 3, 2, 2),
        # Silent: nothing new within a year.
        _history_row("s", "2020-01-01", 1, 1, 1),
        _history_row("s", "2021-01-01", 1, 0, 1),
        # Ambiguous: one new document.
        _history_row("m", "2020-01-01", 1, 1, 1),
        _history_row("m", "2021-01-01", 2, 1, 1),
        # Inactive at T: not a sample at all.
        _history_row("i", "2020-01-01", 1, 0, 1),
        _history_row("i", "2021-01-01", 5, 4, 3),
    ]
    rows = label_rows(history, date(2021, 6, 1))
    by_key = {(r["technology_id"], r["snapshot_date"]): r for r in rows}
    assert by_key[("a", "2020-01-01")]["signal_12m"] == "1"
    assert by_key[("s", "2020-01-01")]["signal_12m"] == "0"
    assert by_key[("m", "2020-01-01")]["signal_12m"] == ""
    assert ("i", "2020-01-01") not in by_key
    # A horizon past the data end is censored, never a negative.
    assert by_key[("a", "2021-01-01")]["outcome_12m"] == "censored"
    assert by_key[("a", "2020-01-01")]["outcome_36m"] == "censored"
    assert {r["label_source"] for r in rows} == {"auto_future_outcome"}


def test_mainstream_is_a_down_weighted_negative():
    history = [
        _history_row("big", "2020-01-01", 40, 20, 5, pct=0.99),
        _history_row("big", "2021-01-01", 80, 40, 9, pct=0.99),
    ]
    row = label_rows(history, date(2022, 1, 1))[0]
    assert row["signal_12m"] == "0"
    assert row["outcome_12m"] == "mainstream"
    assert row["sample_weight_factor"] == 0.5


def test_families_join_parents_duplicates_and_versions():
    log = {
        "families": [
            {
                "groups": [
                    {
                        "canonical": {"concept_id": "a"},
                        "members": [{"concept_id": "b"}],
                    }
                ],
                "held": [
                    {"canonical": {"concept_id": "c"}, "concept_id": "d"}
                ],
                "review": [],
            }
        ]
    }
    found = families("abcdef", [("e", "f"), *plan_pairs(log)])
    assert found["b"] == "a" and found["d"] == "c" and found["f"] == "e"


def _labelled(technology, first_seen, when, label):
    return {
        "technology_id": technology,
        "family_id": technology,
        "snapshot_date": when,
        "horizon_12m_end": add_months(
            date.fromisoformat(when), 12
        ).isoformat(),
        "signal_12m": label,
        "_first": first_seen,
    }


def test_split_keeps_families_apart_and_later_families_later():
    rows = [
        _labelled(
            f"t{i}", f"{2010 + i}-01-01", f"{2010 + i}-06-01", str(i % 2)
        )
        for i in range(10)
    ]
    rows.append(_labelled("t0", "2010-01-01", "2019-06-01", "1"))
    first = {row["family_id"]: row["_first"] for row in rows}
    split = assign_splits(rows, "signal_12m", (0.6, 0.2, 0.2), first)
    parts = {}
    for row in rows:
        parts.setdefault(row["family_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in parts.values())
    order = {"train": 0, "valid": 1, "test": 2}
    ranked = sorted(rows, key=lambda row: first[row["family_id"]])
    assert [order[r["split"]] for r in ranked] == sorted(
        order[r["split"]] for r in ranked
    )
    assert split["embargo"] == "none"


def test_strict_embargo_purges_labels_reaching_the_next_part():
    rows = [
        _labelled(
            f"t{i}", f"{2010 + i}-01-01", f"{2010 + i}-06-01", str(i % 2)
        )
        for i in range(10)
    ]
    # An old technology's late label reaches into valid and test.
    rows.append(_labelled("t0", "2010-01-01", "2019-06-01", "1"))
    first = {row["family_id"]: row["_first"] for row in rows}
    split = assign_splits(
        rows, "signal_12m", (0.6, 0.2, 0.2), first, embargo="strict"
    )
    late = next(
        row
        for row in rows
        if (row["technology_id"], row["snapshot_date"]) == ("t0", "2019-06-01")
    )
    assert late["split"] == "purged"
    for row in rows:
        if row["split"] == "train":
            assert row["horizon_12m_end"] <= split["valid_start"]


def test_feature_selection_drops_empty_constant_and_twin_columns():
    rows = [
        {
            "a": str(i),
            "twin": str(2 * i),
            "flat": "1",
            "empty": "",
            "b": str(i % 3),
        }
        for i in range(20)
    ]
    result = select_features(
        rows, ["a", "twin", "flat", "empty", "b"], 0.99, 0.98
    )
    assert result["selected"] == ["a", "b"]
    assert result["dropped"] == {
        "twin": "correlated with a",
        "flat": "constant",
        "empty": "missing",
    }


def test_config_rejects_bad_fractions(tmp_path):
    override = tmp_path / "bad.json"
    override.write_text(json.dumps({"split": {"fractions": [0.5, 0.5]}}))
    with pytest.raises(ValueError, match="fractions"):
        load_config(override)


def _sample(technology, when, documents):
    nodes = [
        {
            "id": f"Technology:{technology}",
            "type": "Technology",
            "timestamp": when,
            "features": {"burst_score": float(documents)},
        }
    ]
    edges = []
    for index in range(documents):
        document = f"DocumentVersion:{technology}-{index}"
        nodes.append(
            {
                "id": document,
                "type": "DocumentVersion",
                "timestamp": when,
                "source_family": "scholarly",
                "features": {"reliability_tier": 3, "fulltext_available": 1},
            }
        )
        edges.append(
            {
                "source": f"Technology:{technology}",
                "target": document,
                "type": "MENTIONED_IN",
                "timestamp": when,
            }
        )
    return {
        "technology_id": technology,
        "root_id": f"Technology:{technology}",
        "snapshot": when,
        "nodes": nodes,
        "edges": edges,
    }


def test_batched_hgt_matches_single_graphs():
    pytest.importorskip("torch_geometric")
    from lctrend.modeling.training.hgt_model import (
        build_model,
        hgt_data,
        predict_logits,
    )

    scaler = {"burst_score": {"center": 0.0, "scale": 1.0}}
    graphs = [
        hgt_data(
            _sample(f"t{i}", "2020-01-01", i + 1), ["burst_score"], scaler
        )
        for i in range(4)
    ]
    model = build_model(1)
    model.eval()
    single = [model(graph)[0].item() for graph in graphs]
    assert predict_logits(model, graphs, batch_size=3).tolist() == (
        pytest.approx(single, abs=1e-5)
    )


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_pipeline_trains_compares_stacks_and_scores(tmp_path):
    pytest.importorskip("catboost")
    pytest.importorskip("torch_geometric")
    from lctrend.modeling.training.pipeline import train_pipeline

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows, samples = [], []
    for part, count in (("train", 40), ("valid", 16), ("test", 12)):
        for i in range(count):
            technology = f"{part}-{i}"
            label = i % 2
            documents = 1 + 3 * label + i % 2
            when = "2022-01-01"
            rows.append(
                {
                    "technology_id": technology,
                    "technology": technology,
                    "family_id": technology,
                    "snapshot_date": when,
                    "horizon_12m_end": "2023-01-01",
                    "split": part,
                    "signal_12m": str(label),
                    "label_source": "auto_future_outcome",
                    "burst_score": str(documents + (i % 3)),
                    "nb_comentioned_count": str(i % 4),
                }
            )
            samples.append(_sample(technology, when, documents))
    _write(dataset / "training.csv", rows)
    _write(dataset / "scoring.csv", rows[:5])
    history = tmp_path / "history.csv"
    _write(history, rows)
    subgraphs = tmp_path / "subgraphs.jsonl"
    subgraphs.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "target": "signal_12m",
                "features": ["burst_score", "nb_comentioned_count"],
                "split": {"embargo": "none"},
                "inputs": {
                    "history": {"path": str(history)},
                    "subgraphs": {"path": str(subgraphs)},
                },
            }
        ),
        encoding="utf-8",
    )
    config = load_config(
        run="unit",
        catboost={"iterations": 30},
        hgt={"epochs": 3, "patience": 2, "batch_size": 16},
        stacking={"folds": 2},
        minimum_families_per_class={"train": 1, "valid": 1},
    )
    models = tmp_path / "models"
    scores = tmp_path / "scores.csv"
    report = train_pipeline(
        config, dataset_dir=dataset, output_dir=models, scores_path=scores
    )
    assert set(report["models"]) == {"catboost", "hgt", "stacked"}
    for model in report["models"].values():
        test = model["metrics"]["test"]
        assert 0 <= test["pr_auc"] <= 1
        assert "at_valid_f1_threshold" in test
    stacked = report["models"]["stacked"]
    assert stacked["features"][-1] == "hgt_oof_probability"
    assert report["winner"]["model"] in report["models"]
    for name in (
        "catboost.cbm",
        "hgt.pt",
        "hgt_fold0.pt",
        "stacked.cbm",
        "predictions.csv",
    ):
        assert (models / name).exists()
    with scores.open(encoding="utf-8") as stream:
        scored = list(csv.DictReader(stream))
    assert len(scored) == 5
    assert {"p_catboost", "p_hgt", "p_stacked", "probability"} <= set(
        scored[0]
    )


def test_research_rule_and_shuffled_labels(tmp_path):
    pytest.importorskip("catboost")
    import random as _random

    from lctrend.modeling.training.research import catboost_run, rule

    generator = _random.Random(3)

    def rows(part, count):
        result = []
        for i in range(count):
            label = int(i % 3 == 0)
            result.append(
                {
                    "technology_id": f"{part}-{i}",
                    "family_id": f"{part}-{i}",
                    "snapshot_date": "2022-01-01",
                    "split": part,
                    "signal_llm": str(label),
                    # A strong signal and noise.
                    "burst_score": str(label * 3 + generator.random()),
                    "mention_growth_12m": str(generator.random()),
                }
            )
        return result

    study = {
        "target": "signal_llm",
        "parts": {
            "train": rows("train", 90),
            "valid": rows("valid", 45),
            "test": rows("test", 45),
        },
    }
    found = rule(study)
    assert found["rule"] == "+burst_score"
    assert found["test_pr_auc"] > 0.9
    config = load_config(catboost={"iterations": 40})
    features = ["burst_score", "mention_growth_12m"]
    real = catboost_run(study, config, features, 13)
    shuffled = catboost_run(study, config, features, 13, shuffle=True)
    assert real["metrics"]["test"]["pr_auc"] > 0.9
    # Trained on shuffled labels the model loses the signal.
    assert (
        shuffled["metrics"]["test"]["pr_auc"]
        < real["metrics"]["test"]["pr_auc"] - 0.2
    )


def test_platt_calibration_moves_the_prior_back():
    pytest.importorskip("sklearn")
    import numpy as np

    from lctrend.modeling.training.catboost_model import calibrated, fit_platt

    generator = np.random.default_rng(0)
    # 10% positives; a balanced-weight model centres logits on 50/50.
    labels = (generator.random(4000) < 0.1).astype(int)
    logits = np.where(labels == 1, 1.0, -0.2) + generator.normal(0, 1, 4000)
    temperature, intercept = fit_platt(logits, labels)
    probabilities = calibrated(logits, temperature, intercept)
    assert abs(probabilities.mean() - labels.mean()) < 0.01
    assert intercept < 0


def test_top_k_metrics_measure_the_shown_list():
    from lctrend.modeling.training.catboost_model import (
        _metrics,
        top_k_metrics,
    )

    labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    top = top_k_metrics(labels, scores, 3)
    assert top["precision"] == 2 / 3
    assert top["recall"] == 1.0
    # 2/3 of the list against a 20 % positive rate.
    assert round(top["lift"], 6) == round((2 / 3) / 0.2, 6)
    assert top_k_metrics(labels, scores, 100)["k"] == 10
    values = _metrics(labels, scores, [1.0] * 10)
    assert set(values["top_k"]) == {"15", "50", "100"}
    assert values["top_decile"]["k"] == 1
    assert values["log_loss"] > 0
