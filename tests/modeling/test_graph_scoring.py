import csv
import json
from datetime import date

import pytest

from lctrend.graph.temporal import TemporalCorpus
from lctrend.modeling.config import load_config
from lctrend.modeling.labeling.graph_scoring import (
    export_snapshot,
    load_models,
    score_rows,
    scoring_inputs,
)


def _source():
    years = (2018, 2021, 2024, 2025, 2026)
    versions, mentions = [], []
    for year in years:
        versions.append(
            {
                "document_id": f"d{year}",
                "version_id": f"v{year}",
                "document_type": "article",
                "source_family": "scholarly",
                "source_id": "openalex",
                "document_published_at": f"{year}-01-01",
                "version_published_at": f"{year}-01-01",
                "retrieved_at": f"{year}-01-01",
                "extracted": True,
                "extracted_at": f"{year}-01-01",
                "coverage": "full_text",
                "companies": ["Acme"],
            }
        )
        for technology in ("a", "b") if year >= 2024 else ("a",):
            mentions.append(
                {
                    "technology_id": technology,
                    "version_id": f"v{year}",
                    "observed_at": f"{year}-01-01",
                    "mentions": 1,
                    "accepted": 1,
                }
            )
    return {
        "versions": versions,
        "technologies": [
            {"technology_id": "a", "technology": "Alpha"},
            {"technology_id": "b", "technology": "Beta"},
            {"technology_id": "c", "technology": "No documents"},
        ],
        "mentions": mentions,
    }


def test_snapshot_export_gives_every_documented_technology_a_row_and_graph(
    tmp_path,
):
    corpus = TemporalCorpus(_source())
    report = export_snapshot(
        corpus,
        date(2026, 1, 1),
        tmp_path / "history.csv",
        tmp_path / "subgraphs.jsonl",
        include_novelty=False,
    )
    assert report["technologies_with_documents"] == report["subgraphs"] == 2
    rows, samples = scoring_inputs(
        tmp_path / "history.csv", tmp_path / "subgraphs.jsonl"
    )
    assert {row["technology_id"] for row in rows} == {"a", "b"}
    assert {row["snapshot_date"] for row in rows} == {"2026-01-01"}
    assert set(samples) == {("a", "2026-01-01"), ("b", "2026-01-01")}
    # a and b share documents: each is the other's co-mentioned neighbour.
    assert all(row["nb_comentioned_count"] == 1 for row in rows)


def test_a_document_after_the_snapshot_changes_nothing(tmp_path):
    source = _source()
    before = TemporalCorpus(source)
    source["versions"].append(
        {**source["versions"][-1], "document_id": "late", "version_id": "vl"}
        | {
            key: "2026-01-02"
            for key in (
                "document_published_at",
                "version_published_at",
                "retrieved_at",
                "extracted_at",
            )
        }
    )
    source["mentions"].append(
        {
            "technology_id": "b",
            "version_id": "vl",
            "observed_at": "2026-01-02",
            "mentions": 1,
            "accepted": 1,
        }
    )
    after = TemporalCorpus(source)
    exported = []
    for name, corpus in (("before", before), ("after", after)):
        export_snapshot(
            corpus,
            date(2026, 1, 1),
            tmp_path / f"{name}.csv",
            tmp_path / f"{name}.jsonl",
            include_novelty=False,
        )
        exported.append(
            (
                (tmp_path / f"{name}.csv").read_text(encoding="utf-8"),
                (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8"),
            )
        )
    assert exported[0] == exported[1]


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


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_saved_models_reproduce_the_training_scores(tmp_path):
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
            rows.append(
                {
                    "technology_id": technology,
                    "technology": technology,
                    "family_id": technology,
                    "snapshot_date": "2022-01-01",
                    "horizon_12m_end": "2023-01-01",
                    "split": part,
                    "signal_12m": str(label),
                    "label_source": "auto_future_outcome",
                    "burst_score": str(documents + (i % 3)),
                    "nb_comentioned_count": str(i % 4),
                }
            )
            samples.append(_sample(technology, "2022-01-01", documents))
    scoring = rows[:6]
    _write(dataset / "training.csv", rows)
    _write(dataset / "scoring.csv", scoring)
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
    models_dir = tmp_path / "models"
    trained = tmp_path / "scores.csv"
    report = train_pipeline(
        config, dataset_dir=dataset, output_dir=models_dir, scores_path=trained
    )
    with trained.open(encoding="utf-8") as stream:
        expected = {row["technology_id"]: row for row in csv.DictReader(stream)}

    by_key = {
        (sample["technology_id"], sample["snapshot"]): sample
        for sample in samples
    }
    records = score_rows(load_models(models_dir), scoring, by_key)

    assert len(records) == len(scoring)
    for record in records:
        saved = expected[record["technology_id"]]
        for name in ("p_catboost", "p_hgt", "p_stacked", "probability"):
            assert record[name] == pytest.approx(float(saved[name]), abs=1e-5)
        assert record["model"] == report["winner"]["model"]
        assert str(record["signal"]) == saved["signal"]
    probabilities = [record["probability"] for record in records]
    assert probabilities == sorted(probabilities, reverse=True)


def test_a_row_without_subgraph_falls_back_to_catboost(tmp_path):
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    rows = [
        {
            "technology_id": f"t{i}",
            "technology": f"t{i}",
            "snapshot_date": "2022-01-01",
            "burst_score": str(i),
        }
        for i in range(20)
    ]
    model = CatBoostClassifier(iterations=5, verbose=False)
    model.fit([[i] for i in range(20)], [i % 2 for i in range(20)])
    model.set_feature_names(["burst_score"])
    calibration = {
        "temperature_valid_only": 1.0,
        "threshold_valid_f1": 0.5,
        "features": ["burst_score"],
    }
    models = {
        "report": {
            "models": {"catboost": calibration, "stacked": calibration},
            "winner": {"model": "stacked"},
        },
        "catboost": model,
    }
    records = score_rows(models, rows, {})
    assert {record["model"] for record in records} == {"catboost"}
    assert all(record["p_stacked"] == "" for record in records)


def test_only_the_named_key_is_used_even_if_disabled(tmp_path, monkeypatch):
    from pathlib import Path

    from lctrend.modeling.labeling.graph_scoring import only_key

    pool = tmp_path / "keys.json"
    pool.write_text(
        json.dumps(
            {
                "keys": [
                    {"name": "main", "auth_key": "a"},
                    {"name": "Разметка", "auth_key": "b", "enabled": False},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GIGACHAT_KEYS_FILE", str(pool))
    import os

    with only_key("разметка") as name:
        path = Path(os.environ["GIGACHAT_KEYS_FILE"])
        keys = json.loads(path.read_text(encoding="utf-8"))["keys"]
        assert name == "Разметка"
        assert keys == [{"name": "Разметка", "auth_key": "b", "enabled": True}]
    assert os.environ["GIGACHAT_KEYS_FILE"] == str(pool)
    assert not path.exists()
