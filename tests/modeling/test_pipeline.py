import copy
from datetime import date

import pytest

from lctrend.graph.subgraphs import sample_subgraph
from lctrend.graph.temporal import TemporalCorpus
from lctrend.graph.training import build_dataset_rows
from lctrend.modeling.dataset.annotations import (
    build_pilot_queue,
    export_full_history,
    export_pilot_features,
    snapshot_grid,
    write_pilot_queue,
)
from lctrend.modeling.dataset.labels import (
    prepare_labeled_rows,
    require_trainable,
    split_for,
)
from lctrend.modeling.training.catboost_model import (
    explain_catboost,
    train_catboost,
)
from lctrend.modeling.training.hgt_model import (
    build_model,
    explain_hgt,
    fit_scaler,
    hgt_data,
    train_hgt,
)


def _source():
    years = (2010, 2011, 2014, 2020, 2026)
    return {
        "versions": [
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
            for year in years
        ],
        "technologies": [{"technology_id": "t", "technology": "Technique"}],
        "mentions": [
            {
                "technology_id": "t",
                "version_id": f"v{year}",
                "observed_at": f"{year}-01-01",
                "mentions": 1,
                "accepted": 1,
            }
            for year in years
        ],
    }


def test_pilot_has_three_historical_slices_and_no_future_documents():
    corpus = TemporalCorpus(_source())
    rows = build_pilot_queue(corpus, technology_limit=1)
    assert len(rows) == 3
    assert all(row["horizon_12m_end"] <= "2026-01-01" for row in rows)
    assert any(row["horizon_end"] > "2026-01-01" for row in rows)
    assert all("d2026" not in row["recent_document_ids"] for row in rows)
    assert rows[0]["reviewer_signal_36m"] == ""
    assert rows[0]["reviewer_signal_12m"] == ""
    dates = list(snapshot_grid(date(2015, 4, 1)))
    assert date(2004, 1, 1) in dates
    assert date(2005, 7, 1) in dates
    assert date(2015, 4, 1) in dates


def test_model_grid_matches_pilot_dates():
    rows = build_dataset_rows(
        TemporalCorpus(_source()),
        start_year=2013,
        include_novelty=False,
        model_grid=True,
    )
    dates = {row["snapshot_date"] for row in rows}
    assert "2014-07-01" in dates
    assert "2015-04-01" in dates


def test_pilot_feature_export_keeps_labels_empty(tmp_path):
    corpus = TemporalCorpus(_source())
    rows = build_pilot_queue(corpus, technology_limit=1)
    write_pilot_queue(tmp_path / "review.csv", rows)
    output = tmp_path / "features.csv"
    graphs = tmp_path / "graphs.jsonl"
    assert (
        export_pilot_features(
            corpus,
            tmp_path / "review.reviewer_1.csv",
            output,
            graphs,
            include_novelty=False,
        )
        == 3
    )
    import csv
    import json

    with output.open(encoding="utf-8") as stream:
        features = list(csv.DictReader(stream))
    assert all(row["signal_36m"] == "" for row in features)
    assert all(row["signal_12m"] == "" for row in features)
    assert all(row["horizon_end"] for row in features)
    assert len(graphs.read_text().splitlines()) == 3
    sample = json.loads(graphs.read_text().splitlines()[0])
    assert sample["label"] is None
    assert not any(
        name.startswith("embedding_")
        for node in sample["nodes"]
        for name in node["features"]
    )


def test_full_export_includes_single_document_technology_and_all_slices(
    tmp_path,
):
    source = _source()
    source["technologies"].append(
        {"technology_id": "single", "technology": "One paper"}
    )
    source["mentions"].append(
        {
            "technology_id": "single",
            "version_id": "v2026",
            "observed_at": "2026-01-01",
            "mentions": 1,
            "accepted": 1,
        }
    )
    report = export_full_history(
        TemporalCorpus(source),
        tmp_path / "full.csv",
        tmp_path / "review.csv",
        tmp_path / "graphs.jsonl",
        include_novelty=False,
    )
    import csv
    import json

    with (tmp_path / "full.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with (tmp_path / "review.reviewer_1.csv").open(encoding="utf-8") as stream:
        review = list(csv.DictReader(stream))
    samples = [
        json.loads(line)
        for line in (tmp_path / "graphs.jsonl").read_text().splitlines()
    ]
    assert report["rows"] == len(rows) == len(review) == len(samples)
    assert len(rows) > 3
    assert report["technologies_with_snapshots"] == 2
    assert any(row["technology_id"] == "single" for row in rows)
    assert any(row["snapshot_date"] == "2026-01-01" for row in rows)
    assert all(row["signal_12m"] == row["signal_36m"] == "" for row in rows)
    assert all(sample["label"] is None for sample in samples)
    assert all(row["reviewer_signal_12m"] == "" for row in review)


def _review(label="1", note="doi:10.1234/example"):
    return {
        "technology_id": "t",
        "snapshot_date": "2013-01-01",
        "horizon_end": "2016-01-01",
        "family_id": "family-t",
        "reviewer_signal_36m": label,
        "reviewer_trend_36m": "0",
        "evidence_notes": note,
    }


def test_only_consensus_or_audited_decision_produces_label():
    source = [
        {
            "technology_id": "t",
            "snapshot_date": "2013-01-01",
            "horizon_end": "2016-01-01",
            "document_count": "2",
        }
    ]
    rows = prepare_labeled_rows(
        source, [_review()], [_review()], data_end="2017-01-01"
    )
    assert rows[0]["signal_36m"] == 1
    assert rows[0]["split"] == "train"
    disputed = prepare_labeled_rows(
        source, [_review()], [_review("0")], data_end="2017-01-01"
    )
    assert disputed[0]["signal_36m"] is None
    censored = prepare_labeled_rows(
        source, [_review()], [_review()], data_end="2015-01-01"
    )
    assert censored[0]["signal_36m"] is None
    with pytest.raises(ValueError, match="evidence notes"):
        prepare_labeled_rows(
            source, [_review(note="")], [_review()], data_end="2017-01-01"
        )


def test_horizons_are_independent_and_recent_three_years_are_censored():
    source = [
        {
            "technology_id": "t",
            "snapshot_date": "2022-01-01",
            "horizon_12m_end": "2023-01-01",
            "horizon_end": "2025-01-01",
        },
        {
            "technology_id": "new",
            "snapshot_date": "2024-01-01",
            "horizon_12m_end": "2025-01-01",
            "horizon_end": "2027-01-01",
        },
    ]
    reviews = [
        {
            **row,
            "family_id": row["technology_id"],
            "reviewer_signal_12m": "0",
            "reviewer_signal_36m": "1",
            "evidence_notes": "Independent dated evidence checked",
        }
        for row in source
    ]
    result = prepare_labeled_rows(
        source, reviews, reviews, data_end="2026-09-01"
    )
    assert result[0]["signal_12m"] == 0
    assert result[0]["signal_36m"] == 1
    assert result[0]["split_12m"] == "valid"
    assert result[1]["signal_12m"] == 0
    assert result[1]["signal_36m"] is None
    assert result[1]["split_12m"] == "test"


def test_family_folds_keep_all_dates_of_a_technology_together():
    source = [
        {
            "technology_id": "t",
            "snapshot_date": when,
            "horizon_12m_end": end_12,
            "horizon_end": end_36,
        }
        for when, end_12, end_36 in (
            ("2019-01-01", "2020-01-01", "2022-01-01"),
            ("2021-01-01", "2022-01-01", "2024-01-01"),
        )
    ]
    review = [
        {
            **row,
            "family_id": "same-family",
            "reviewer_signal_12m": "1",
            "reviewer_signal_36m": "1",
            "evidence_notes": "Independent dated evidence checked",
        }
        for row in source
    ]
    result = prepare_labeled_rows(
        source, review, review, data_end="2026-09-01"
    )
    assert result[0]["family_fold"] == result[1]["family_fold"]
    for fold in range(5):
        assert split_for(result[0], strategy="family", fold=fold) == split_for(
            result[1], strategy="family", fold=fold
        )


def test_family_split_can_train_without_temporal_windows():
    rows = [
        {
            "technology_id": str(index),
            "family_id": str(index),
            "family_fold": 0 if index < 2 else 1,
            "snapshot_date": "2020-01-01",
            "horizon_12m_end": "2021-01-01",
            "horizon_end": "2023-01-01",
            "signal_12m": index % 2,
            "label_source": "expert_consensus",
        }
        for index in range(4)
    ]
    require_trainable(
        rows, 1, 1, target="signal_12m", strategy="family", fold=0
    )


def test_dynamic_cohort_split_approximates_80_10_10_by_first_seen_year():
    source = []
    for index in range(10):
        first_year = 2019 if index < 8 else 2020 if index == 8 else 2021
        for year in (first_year, first_year + 1):
            source.append(
                {
                    "technology_id": f"t{index}",
                    "snapshot_date": f"{year}-01-01",
                    "first_seen_date": f"{first_year}-01-01",
                    "horizon_12m_end": f"{year + 1}-01-01",
                    "horizon_end": f"{year + 3}-01-01",
                }
            )
    reviews = [{**row, "family_id": row["technology_id"]} for row in source]
    result = prepare_labeled_rows(
        source, reviews, reviews, data_end="2026-09-01"
    )
    assert [
        sum(row["cohort_split"] == part for row in result)
        for part in ("train", "valid", "test")
    ] == [16, 2, 2]
    assert {split_for(row, strategy="cohort") for row in result[-2:]} == {
        "test"
    }


def test_classifier_refuses_one_class():
    with pytest.raises(ValueError, match="positive and negative"):
        require_trainable(
            [
                {
                    "technology_id": "t",
                    "family_id": "t",
                    "split": "train",
                    "signal_36m": 1,
                    "snapshot_date": "2013-01-01",
                    "horizon_end": "2016-01-01",
                    "label_source": "expert_consensus",
                }
            ]
        )


def test_hgt_converts_reverse_mentions_and_root_receives_messages():
    corpus = TemporalCorpus(_source())
    sample = sample_subgraph(
        corpus.view(date(2022, 1, 1)), "t", features={"document_count": 4}
    )
    data = hgt_data(
        sample,
        ["document_count"],
        fit_scaler([{"document_count": 4}], ["document_count"]),
    )
    assert (
        data[("Document", "old_mentions", "Technology")].edge_index.size(1) > 0
    )
    assert build_model(1)(data).shape == (2,)


def test_catboost_training_and_probability_explanation(tmp_path):
    pytest.importorskip("catboost")
    train = [
        {
            "technology_id": f"train-{i}",
            "family_id": f"train-{i}",
            "snapshot_date": "2013-01-01",
            "horizon_end": "2016-01-01",
            "label_source": "expert_consensus",
            "split": "train",
            "signal_36m": i % 2,
            "max_maturity_rank": i % 2,
            "document_count_snapshot_pct": i / 6,
        }
        for i in range(6)
    ]
    valid = [
        {
            "technology_id": f"valid-{i}",
            "family_id": f"valid-{i}",
            "snapshot_date": "2017-01-01",
            "horizon_end": "2020-01-01",
            "label_source": "expert_consensus",
            "split": "valid",
            "signal_36m": i % 2,
            "max_maturity_rank": i % 2,
            "document_count_snapshot_pct": i / 4,
        }
        for i in range(4)
    ]
    report = train_catboost(
        train + valid,
        tmp_path,
        iterations=25,
        min_train_families_per_class=1,
        min_valid_families_per_class=1,
    )
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(tmp_path / "catboost_signal.cbm"))
    explanation = explain_catboost(model, valid[0], report)
    assert 0 <= explanation["probability"] <= 1
    assert explanation["features"]


def test_hgt_training_and_entity_explanations(tmp_path):
    pytest.importorskip("torch_geometric")
    corpus = TemporalCorpus(_source())
    rows, samples = [], []
    for part, when, count in (
        ("train", date(2013, 1, 1), 4),
        ("valid", date(2017, 1, 1), 2),
    ):
        for index in range(count):
            label = index % 2
            technology_id = f"{part}-{index}"
            row = {
                "technology_id": technology_id,
                "family_id": technology_id,
                "snapshot_date": when.isoformat(),
                "horizon_end": date(when.year + 3, 1, 1).isoformat(),
                "split": part,
                "label_source": "expert_consensus",
                "signal_36m": label,
                "max_maturity_rank": label,
                "document_count_snapshot_pct": label / 2,
            }
            sample = sample_subgraph(
                corpus.view(when),
                "t",
                features={
                    "max_maturity_rank": label,
                    "document_count_snapshot_pct": label / 2,
                },
            )
            sample = copy.deepcopy(sample)
            sample["technology_id"] = technology_id
            sample["root_id"] = "Technology:" + technology_id
            for node in sample["nodes"]:
                if node["id"] == "Technology:t":
                    node["id"] = sample["root_id"]
            for edge in sample["edges"]:
                for key in ("source", "target"):
                    if edge[key] == "Technology:t":
                        edge[key] = sample["root_id"]
            rows.append(row)
            samples.append(sample)
    report = train_hgt(
        rows,
        samples,
        tmp_path,
        epochs=1,
        min_train_families_per_class=1,
        min_valid_families_per_class=1,
    )
    import torch

    model = build_model(len(report["features"]))
    model.load_state_dict(
        torch.load(tmp_path / "hgt_signal.pt", weights_only=True)
    )
    explanation = explain_hgt(model, samples[-1], report, top_k=2)
    assert 0 <= explanation["probability"] <= 1
    assert explanation["entities"]
