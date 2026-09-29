import csv
import hashlib
import json
import zipfile

import pytest

from lctrend.modeling.dataset.neighbors import (
    NEIGHBOR_FEATURES,
    enrich_file,
    neighbor_features,
    neighbor_groups,
)
from lctrend.modeling.storage import RunLayout, write_manifest

SNAPSHOT = "2020-01-01"


def _node(identifier, kind):
    return {
        "id": identifier,
        "type": kind,
        "features": {},
        "timestamp": SNAPSHOT,
    }


def _sample():
    """Root r: parent p, co-mentioned c via document d, company, university.

    x is mentioned by another document only and is not a co-mention.
    """
    nodes = [
        _node("Technology:r", "Technology"),
        _node("Technology:p", "Technology"),
        _node("Technology:c", "Technology"),
        _node("Technology:x", "Technology"),
        _node("DocumentVersion:d", "DocumentVersion"),
        _node("DocumentVersion:e", "DocumentVersion"),
        _node("Company:acme", "Company"),
        _node("University:mit", "University"),
    ]
    edges = [
        ("Technology:r", "Technology:p", "SUBTECHNOLOGY_OF"),
        ("Technology:r", "DocumentVersion:d", "MENTIONED_IN"),
        ("Technology:c", "DocumentVersion:d", "MENTIONED_IN"),
        ("Technology:x", "DocumentVersion:e", "MENTIONED_IN"),
        ("DocumentVersion:d", "Company:acme", "ASSOCIATED_WITH"),
        ("DocumentVersion:d", "University:mit", "ASSOCIATED_WITH"),
    ]
    return {
        "technology_id": "r",
        "root_id": "Technology:r",
        "snapshot": SNAPSHOT,
        "nodes": nodes,
        "edges": [
            {"source": s, "target": t, "type": kind, "timestamp": SNAPSHOT}
            for s, t, kind in edges
        ],
    }


def _row(technology, growth, share, date=SNAPSHOT):
    return {
        "technology_id": technology,
        "snapshot_date": date,
        "mention_growth_12m": str(growth),
        "documents_last_year_snapshot_pct": str(share),
    }


def test_neighbor_groups_separate_typed_links_from_co_mentions():
    assert neighbor_groups(_sample()) == {
        "related": {"p"},
        "comentioned": {"c"},
    }


def test_neighbor_features_join_rows_of_the_same_snapshot():
    rows = {
        ("p", SNAPSHOT): _row("p", 0.5, 0.9),
        ("c", SNAPSHOT): _row("c", -0.2, 0.3),
        # A later row of the same technology must not leak in.
        ("c", "2021-01-01"): _row("c", 9.0, 1.0, "2021-01-01"),
    }
    values = neighbor_features(_sample(), rows)
    assert set(values) == set(NEIGHBOR_FEATURES)
    assert values["nb_related_count"] == 1
    assert values["nb_related_mention_growth_12m_mean"] == 0.5
    assert values["nb_comentioned_documents_last_year_snapshot_pct_max"] == 0.3
    assert values["nb_growing_share"] == 0.5
    assert values["nb_organization_count"] == 2
    assert values["nb_company_share"] == 0.5
    # A value no neighbour has is missing, not zero.
    assert values["nb_related_new_author_rate_mean"] is None


def test_no_neighbours_give_missing_means_and_zero_counts():
    sample = _sample()
    sample["edges"] = []
    values = neighbor_features(sample, {})
    assert values["nb_comentioned_count"] == 0
    assert values["nb_comentioned_burst_score_mean"] is None
    assert values["nb_growing_share"] is None


def test_enrich_file_reads_subgraphs_from_zip(tmp_path):
    dataset = tmp_path / "history.csv"
    rows = [
        {**_row("r", 0.1, 0.5), "technology": "root"},
        {**_row("p", 0.5, 0.9), "technology": "parent"},
        {**_row("c", -0.2, 0.3), "technology": "co"},
    ]
    with dataset.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    archive = tmp_path / "subgraphs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("s.jsonl", json.dumps(_sample()) + "\n")
    output = tmp_path / "out.csv"
    summary = enrich_file(dataset, archive, output)
    assert summary["enriched"] == 1
    with output.open(encoding="utf-8") as stream:
        result = {row["technology_id"]: row for row in csv.DictReader(stream)}
    assert float(result["r"]["nb_related_mention_growth_12m_max"]) == 0.5
    assert result["p"]["nb_related_count"] == ""


def test_hgt_neighbour_technologies_take_their_own_rows():
    pytest.importorskip("torch_geometric")
    from lctrend.modeling.training.hgt_model import hgt_data

    names = ["mention_growth_12m"]
    scaler = {"mention_growth_12m": {"center": 0.0, "scale": 1.0}}
    sample = _sample()
    sample["nodes"][0]["features"] = {"mention_growth_12m": 0.1}
    rows = {("p", SNAPSHOT): _row("p", 0.5, 0.9)}
    blank = hgt_data(sample, names, scaler)
    filled = hgt_data(sample, names, scaler, rows)
    ids = filled["Technology"].node_ids
    parent = ids.index("Technology:p")
    # Value, then its missing flag.
    assert blank["Technology"].x[parent].tolist() == [0.0, 1.0]
    assert filled["Technology"].x[parent].tolist() == [0.5, 0.0]


def test_catboost_scores_each_technology_at_its_latest_snapshot(tmp_path):
    pytest.importorskip("catboost")
    from lctrend.modeling.labeling.scoring import score_catboost_file
    from lctrend.modeling.training.catboost_model import train_catboost

    def labeled(part, when, count):
        return [
            {
                "technology_id": f"{part}-{i}",
                "family_id": f"{part}-{i}",
                "snapshot_date": when,
                "horizon_end": when.replace(when[:4], str(int(when[:4]) + 3)),
                "label_source": "expert_consensus",
                "split": part,
                "signal_36m": i % 2,
                "max_maturity_rank": i % 2,
                "document_count_snapshot_pct": i / count,
                "nb_related_count": i,
            }
            for i in range(count)
        ]

    rows = labeled("train", "2013-01-01", 6)
    rows += labeled("valid", "2017-01-01", 4)
    report = train_catboost(
        rows,
        tmp_path,
        iterations=25,
        min_train_families_per_class=1,
        min_valid_families_per_class=1,
    )
    assert "nb_related_count" in report["features"]
    dataset = tmp_path / "all.csv"
    history = [
        {"technology_id": "t", "technology": "T", "snapshot_date": d}
        | {name: "1" for name in report["features"]}
        for d in ("2019-01-01", "2024-01-01")
    ]
    with dataset.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    output = tmp_path / "scores.csv"
    summary = score_catboost_file(tmp_path, dataset, output)
    with output.open(encoding="utf-8") as stream:
        scored = list(csv.DictReader(stream))
    assert summary["technologies"] == 1
    assert scored[0]["snapshot_date"] == "2024-01-01"
    assert 0 <= float(scored[0]["probability"]) <= 1


def test_manifest_records_input_digest(tmp_path):
    source = tmp_path / "in.csv"
    source.write_bytes(b"a\n1\n")
    output = tmp_path / "out.csv"
    manifest = write_manifest(output, "dataset.test", [source])
    assert (
        manifest["inputs"][0]["sha256"]
        == hashlib.sha256(b"a\n1\n").hexdigest()
    )
    assert (tmp_path / "out.csv.manifest.json").exists()
    layout = RunLayout.at("r1", base=tmp_path)
    assert layout.labeling == tmp_path / "r1" / "labeling"


def test_graph_rows_join_model_scores_and_llm_labels():
    from lctrend.modeling.labeling.graph_labels import PROPERTIES, graph_rows

    scores = [
        {
            "technology_id": "a",
            "probability": "0.8",
            "signal": "1",
            "model": "stacked",
            "snapshot_date": "2026-07-01",
        },
        {
            "technology_id": "b",
            "probability": "",
            "signal": "",
            "model": "stacked",
            "snapshot_date": "2026-07-01",
        },
    ]
    answers = {
        "a": {
            "verdict": "niche",
            "is_technology": True,
            "rationale": "r",
            "model": "m",
            "years": [
                {"year": 2026, "score": 0.2, "hype": 0.3, "maturity": 0.4},
                {"year": 2025, "score": 0.1, "hype": 0.2, "maturity": 0.3},
            ],
        },
        "c": {
            "verdict": "junk",
            "is_technology": False,
            "rationale": "x",
            "model": "m",
            "years": [],
        },
    }
    rows = {
        row["id"]: row["props"] for row in graph_rows(scores, answers, "t")
    }
    assert rows["a"]["signal_probability"] == 0.8 and rows["a"]["signal_flag"]
    # The latest year stands for the technology.
    assert rows["a"]["llm_score"] == 0.2
    assert json.loads(rows["a"]["llm_years"])[0]["year"] == 2025
    # No score, no LLM answer: nothing to write for b.
    assert "b" not in rows
    assert rows["c"]["llm_is_technology"] is False
    assert "llm_score" not in rows["c"]
    assert set(rows["a"]) <= set(PROPERTIES)
