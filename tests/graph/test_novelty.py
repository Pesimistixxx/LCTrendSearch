from datetime import date

import pytest

from lctrend.graph.novelty import NOVELTY_FIELDS, novelty_features
from lctrend.graph.temporal import TemporalCorpus


def corpus_data():
    def version(key, when, domains):
        return {
            "document_id": key,
            "version_id": key + "v",
            "document_type": "article",
            "version_published_at": when,
            "retrieved_at": when,
            "domains": domains,
        }

    def mention(technology, document, when):
        return {
            "technology_id": technology,
            "version_id": document + "v",
            "observed_at": when,
            "mentions": 1,
        }

    return {
        "versions": [
            version("ab", "2018-01-01", ["ai"]),
            version("bc", "2019-06-01", ["bio"]),
        ],
        "mentions": [
            mention("a", "ab", "2018-01-01"),
            mention("b", "ab", "2018-01-01"),
            mention("b", "bc", "2019-06-01"),
            mention("c", "bc", "2019-06-01"),
        ],
        "technologies": [
            {
                "technology_id": "a",
                "embedding": [1, 0],
                "embedding_observed_at": "2018-01-01",
            },
            {
                "technology_id": "b",
                "embedding": [0, 1],
                "embedding_observed_at": "2018-01-01",
            },
            {
                "technology_id": "c",
                "embedding": [1, 0],
                "embedding_observed_at": "2019-06-01",
            },
        ],
        "relations": [
            {
                "technology_id": "a",
                "relation": "SOLVES",
                "target_id": "task",
                "target_kind": "Task",
                "observed_at": "2018-01-01",
                "version_id": "abv",
            },
            {
                "technology_id": "b",
                "relation": "SOLVES",
                "target_id": "task",
                "target_kind": "Task",
                "observed_at": "2018-01-01",
                "version_id": "abv",
            },
        ],
    }


def test_snapshot_graph_has_path_metrics_and_calendar_deltas():
    rows = novelty_features(
        TemporalCorpus(corpus_data()).view(date(2020, 1, 1))
    )
    assert rows["b"]["degree"] == 2
    assert rows["b"]["degree_delta_12m"] == 1
    assert rows["a"]["degree_delta_12m"] == 0
    assert rows["b"]["betweenness"] == pytest.approx(1.0)
    assert rows["a"]["betweenness"] == 0
    assert rows["b"]["k_core"] == 1
    assert sum(row["pagerank"] for row in rows.values()) == pytest.approx(1.0)
    assert rows["b"]["structural_hole_score"] == pytest.approx(0.5)
    assert rows["b"]["new_domain_pair_count"] == 1
    assert rows["b"]["neighbor_domain_entropy"] == pytest.approx(0.69314718)
    assert rows["c"]["nearest_known_distance"] == pytest.approx(0.0)
    assert rows["a"]["task_combination_novelty"] == 0.0
    assert set(rows["a"]) == set(NOVELTY_FIELDS)


def test_future_documents_relations_and_vectors_do_not_change_features():
    data = corpus_data()
    before = novelty_features(TemporalCorpus(data).view(date(2020, 1, 1)))
    data["versions"].append(
        {
            "document_id": "future",
            "version_id": "futurev",
            "version_published_at": "2021-01-01",
        }
    )
    data["mentions"].extend(
        [
            {
                "technology_id": technology,
                "version_id": "futurev",
                "observed_at": "2021-01-01",
                "mentions": 100,
            }
            for technology in ("a", "c", "future")
        ]
    )
    data["technologies"].append(
        {
            "technology_id": "future",
            "embedding": [-1, 0],
            "embedding_observed_at": "2021-01-01",
        }
    )
    data["relations"].append(
        {
            "technology_id": "a",
            "target_id": "future",
            "relation": "SUBTECHNOLOGY_OF",
            "version_id": "futurev",
            "observed_at": "2019-01-01",
        }
    )
    after = novelty_features(TemporalCorpus(data).view(date(2020, 1, 1)))
    assert after == before


def test_missing_embedding_does_not_invent_semantic_novelty():
    data = corpus_data()
    for row in data["technologies"]:
        row.pop("embedding_observed_at")
    rows = novelty_features(TemporalCorpus(data).view(date(2020, 1, 1)))
    assert rows["a"]["semantic_novelty"] is None
    assert rows["a"]["cluster_centroid_distance"] is None
    assert rows["a"]["taxonomy_depth"] is None


def test_same_day_concept_cannot_be_semantic_reference():
    rows = novelty_features(
        TemporalCorpus(corpus_data()).view(date(2018, 1, 1))
    )
    assert rows["a"]["nearest_known_distance"] is None
    assert rows["b"]["nearest_known_distance"] is None


def test_sampled_betweenness_is_repeatable():
    snapshot = TemporalCorpus(corpus_data()).view(date(2020, 1, 1))
    config = {"graph": {"betweenness_samples": 1, "seed": 2}}
    assert novelty_features(snapshot, config) == novelty_features(
        snapshot, config
    )
