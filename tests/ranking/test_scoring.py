"""Transparent TOP-15: candidate rule at T, weighted z-score, explanation
and the precision@K backtest against random samples.
"""

import copy
from datetime import date

import pytest

from lctrend.core.config import load_catalog
from lctrend.graph.temporal import TemporalCorpus
from lctrend.graph.training import dataset_feature_fields
from lctrend.ranking.scoring import (
    backtest,
    evidence_quotes,
    rank_snapshot,
    score_rows,
)

T = date(2021, 1, 1)


def corpus_data(technologies):
    """technologies: {id: (label, [(published, group), ...])}."""
    data = {"versions": [], "mentions": [], "technologies": []}
    for technology_id, (label, documents) in technologies.items():
        data["technologies"].append(
            {"technology_id": technology_id, "technology": label}
        )
        for index, (published, group) in enumerate(documents):
            document = f"{technology_id}-{index}"
            data["versions"].append(
                {
                    "document_id": document,
                    "version_id": document + "-v1",
                    "document_type": "article",
                    "source_family": "scholarly",
                    "source_id": "source:openalex",
                    "document_published_at": published,
                    "version_published_at": published,
                    "retrieved_at": "2026-09-20",
                    "independence_group": group,
                    "title": f"Paper {document}",
                    "url": f"https://example.org/{document}",
                    "extracted": True,
                }
            )
            data["mentions"].append(
                {
                    "technology_id": technology_id,
                    "version_id": document + "-v1",
                    "observed_at": published,
                    "mentions": 1,
                    "accepted": 1,
                }
            )
    return data


def pool():
    return corpus_data(
        {
            "fresh": (
                "Sodium solid-state cells",
                [
                    ("2019-06-01", "a"),
                    ("2020-03-01", "b"),
                    ("2020-09-01", "c"),
                    ("2020-11-01", "d"),
                ],
            ),
            "steady": (
                "Liquid neural networks",
                [("2019-02-01", "e"), ("2019-09-01", "f")],
            ),
            "old": (
                "Relational databases",
                [("2010-01-01", "g"), ("2020-05-01", "h")],
            ),
            "lonely": (
                "Single lab trick",
                [
                    ("2020-01-01", "i"),
                    ("2020-06-01", "i"),
                    ("2020-10-01", "i"),
                ],
            ),
            "generic": (
                "Machine learning",
                [("2019-06-01", "j"), ("2020-06-01", "k")],
            ),
            "sold": (
                "Commercial widget",
                [("2019-06-01", "l"), ("2020-06-01", "m")],
            ),
        }
    ) | {
        "maturity": [
            {
                "technology_id": "sold",
                "version_id": "sold-1-v1",
                "observed_at": "2020-06-01",
                "stage_rank": 5,
            }
        ]
    }


def config(**score):
    value = copy.deepcopy(load_catalog("ranking"))
    value["include_novelty"] = False
    value["score"].update(score)
    return value


def test_shipped_weights_name_export_features_columns():
    weights = load_catalog("ranking")["score"]["weights"]
    assert weights
    assert set(weights) <= set(dataset_feature_fields())
    assert set(load_catalog("ranking")["score"]["labels"]) == set(weights)


def test_candidate_rule_at_t_explains_every_rejection():
    ranking = rank_snapshot(TemporalCorpus(pool()), T, config())
    assert {item["technology_id"] for item in ranking.candidates} == {
        "fresh",
        "steady",
    }
    rejected = {item["technology_id"]: item for item in ranking.rejected}
    assert rejected["old"]["category"] == "mature"
    assert rejected["sold"]["category"] == "mature"
    assert rejected["generic"]["category"] == "standard"
    assert rejected["lonely"]["category"] == "noise"
    assert "независим" in rejected["lonely"]["reason"]


def test_unknown_maturity_passes_the_candidate_rule():
    ranking = rank_snapshot(TemporalCorpus(pool()), T, config())
    fresh = next(
        item for item in ranking.candidates if item["technology_id"] == "fresh"
    )
    assert fresh["row"]["max_maturity_rank"] is None


def test_score_is_weight_times_z_with_missing_as_zero():
    rows = [
        {"technology_id": "a", "x": 1.0, "y": 5.0},
        {"technology_id": "b", "x": 3.0, "y": 5.0},
        {"technology_id": "c", "x": None, "y": 5.0},
    ]
    scored = score_rows(rows, {"x": 2.0, "y": 1.0}, z_clip=3.0)
    by_id = {item["technology_id"]: item for item in scored}
    assert by_id["a"]["contributions"] == {"x": -2.0, "y": 0.0}
    assert by_id["b"]["contributions"] == {"x": 2.0, "y": 0.0}
    assert by_id["c"]["contributions"] == {"x": 0.0, "y": 0.0}
    assert [item["technology_id"] for item in scored] == ["b", "c", "a"]
    assert by_id["b"]["raw_score"] == 2.0
    assert 0.5 < by_id["b"]["score"] < 1 and by_id["c"]["score"] == 0.5


def test_weights_come_from_the_configuration():
    corpus = TemporalCorpus(pool())
    by_documents = rank_snapshot(
        corpus, T, config(weights={"document_count": 1.0})
    )
    assert by_documents.candidates[0]["technology_id"] == "fresh"
    reversed_ = rank_snapshot(
        corpus, T, config(weights={"document_count": -1.0})
    )
    assert reversed_.candidates[0]["technology_id"] == "steady"
    top = by_documents.candidates[0]
    assert top["contributions"] == {"document_count": pytest.approx(1.0)}


def test_evidence_quotes_come_from_accepted_assertions_known_at_t():
    data = pool()
    data["assertions"] = [
        {
            "technology_id": "fresh",
            "assertion_id": f"a{index}",
            "version_id": version,
            "observed_at": observed,
            "status": status,
            "quote": f"quote {index}",
        }
        for index, (version, observed, status) in enumerate(
            [
                ("fresh-0-v1", "2019-06-01", "accepted"),
                ("fresh-1-v1", "2020-03-01", "needs_review"),
                ("fresh-2-v1", "2020-09-01", "accepted"),
                ("fresh-2-v1", "2020-09-01", "accepted"),
                ("fresh-3-v1", "2020-11-01", "accepted"),
                ("fresh-3-v1", "2022-01-01", "accepted"),
            ]
        )
    ]
    corpus = TemporalCorpus(data)
    view = corpus.view(T).technologies["fresh"]
    quotes = evidence_quotes(corpus, view, 3)
    assert [item["text"] for item in quotes] == [
        "quote 4",
        "quote 2",
        "quote 0",
    ]
    assert quotes[0] == {
        "text": "quote 4",
        "title": "Paper fresh-3",
        "date": "2020-11-01",
        "url": "https://example.org/fresh-3",
    }


def test_explanation_lists_the_three_largest_contributions():
    ranking = rank_snapshot(
        TemporalCorpus(pool()),
        T,
        config(
            weights={
                "document_count": 1.0,
                "mention_count": 0.5,
                "documents_last_year": 0.25,
                "independence_group_diversity": 2.0,
            }
        ),
    )
    features = ranking.candidates[0]["top_features"]
    assert [item["feature"] for item in features] == [
        "independence_group_diversity",
        "document_count",
        "mention_count",
    ]
    assert features[0]["contribution"] == pytest.approx(2.0)


def test_later_data_cannot_change_the_ranking_at_t():
    before = rank_snapshot(TemporalCorpus(pool()), T, config())
    data = pool()
    late = corpus_data(
        {
            "steady": (
                "Liquid neural networks",
                [
                    (f"2022-0{month}-01", f"late{month}")
                    for month in range(1, 8)
                ],
            )
        }
    )
    for row in late["versions"]:
        row["document_id"] += "-late"
        row["version_id"] = row["document_id"] + "-v1"
    for row, version in zip(late["mentions"], late["versions"]):
        row["version_id"] = version["version_id"]
    data["versions"] += late["versions"]
    data["mentions"] += late["mentions"]
    after = rank_snapshot(TemporalCorpus(data), T, config())
    assert [
        (item["technology_id"], item["raw_score"])
        for item in before.candidates
    ] == [
        (item["technology_id"], item["raw_score"]) for item in after.candidates
    ]


def growth_pool():
    data = pool()
    extra = corpus_data(
        {
            "fresh": (
                "Sodium solid-state cells",
                [(f"2021-0{month}-01", f"g{month}") for month in range(2, 9)],
            ),
            "tail": ("Tail", [("2023-06-01", "tail")]),
        }
    )
    for row in extra["versions"]:
        row["document_id"] += "-future"
        row["version_id"] = row["document_id"] + "-v1"
    for row, version in zip(extra["mentions"], extra["versions"]):
        row["version_id"] = version["version_id"]
    data["versions"] += extra["versions"]
    data["mentions"] += extra["mentions"]
    return data


def test_backtest_reports_precision_at_k_against_random_samples():
    corpus = TemporalCorpus(growth_pool())
    settings = config(weights={"document_count": 1.0})
    settings["top_k"] = 1
    result = backtest(corpus, T, settings)
    assert result["snapshot"] == "2021-01-01"
    assert result["horizon_end"] == "2023-01-01"
    assert result["horizon_complete"] is True
    assert result["k"] == 1 and result["candidates"] == 2
    assert [item["technology_id"] for item in result["top"]] == ["fresh"]
    assert result["top"][0]["grew"] is True
    assert result["top"][0]["future_documents"] == 7
    assert result["precision_at_k"] == 1.0
    assert result["base_rate"] == 0.5
    random = result["random"]
    assert random["trials"] == settings["backtest"]["random_trials"]
    assert 0.35 < random["mean_precision"] < 0.65
    assert 0.35 < random["p_value"] < 0.65


def test_backtest_marks_an_unobserved_horizon():
    result = backtest(TemporalCorpus(pool()), date(2020, 6, 1), config())
    assert result["horizon_complete"] is False
    assert result["warnings"]
