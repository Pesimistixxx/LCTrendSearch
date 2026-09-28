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
    # taxonomy_depth was a copy of taxonomy_level and is no longer a column.
    assert rows["a"]["taxonomy_level"] is None


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


def emerging_cluster():
    """Three established terms near [1, 0], five new ones near [0, 1]."""
    versions, mentions, technologies = [], [], []
    for index in range(8):
        new = index >= 3
        when = f"2019-{index + 3:02d}-01" if new else "2015-01-01"
        key = f"t{index}"
        versions.append(
            {
                "document_id": key,
                "version_id": key + "v",
                "document_type": "article",
                "version_published_at": when,
                "retrieved_at": when,
            }
        )
        mentions.append(
            {
                "technology_id": key,
                "version_id": key + "v",
                "observed_at": when,
                "mentions": 1,
            }
        )
        offset = 0.01 * index
        technologies.append(
            {
                "technology_id": key,
                "embedding": [offset, 1.0] if new else [1.0, offset],
                "embedding_observed_at": when,
            }
        )
    return {
        "versions": versions,
        "mentions": mentions,
        "technologies": technologies,
        "relations": [],
    }


def test_an_emerging_cluster_keeps_its_documented_semantic_novelty():
    rows = novelty_features(
        TemporalCorpus(emerging_cluster()).view(date(2020, 1, 1))
    )
    for key in ("t3", "t4", "t5", "t6", "t7"):
        # 1 - cosine to the nearest concept known a year before T.
        assert rows[key]["semantic_novelty"] > 0.5
        # The distance to any earlier concept is a different metric:
        # the cluster's own members are close.
        assert rows[key]["nearest_known_distance"] < 0.01
    assert rows["t0"]["semantic_novelty"] < 0.01


def test_novelty_fields_have_no_duplicate_columns():
    duplicates = {
        "nearest_known_technology_distance",
        "semantic_outlier_score",
        "new_taxonomy_branch",
        "taxonomy_depth",
    }
    assert not duplicates & set(NOVELTY_FIELDS)
