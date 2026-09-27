"""TaxoGen-style taxonomy on synthetic, well-separated label vectors."""

import asyncio
from datetime import date

import numpy as np

from lctrend.core.config import load_catalog
from lctrend.graph.store import GraphStore
from lctrend.graph.training import build_feature_rows
from lctrend.taxonomy import (
    TaxonomyConcept,
    build_taxonomy,
    taxonomy_features,
)

OLD = date(2022, 1, 1)
NEW = date(2025, 6, 1)


def vector(axis, noise, rng, dimensions=8):
    value = np.zeros(dimensions)
    value[axis] = 1.0
    return list(value + rng.normal(0, noise, dimensions))


def corpus():
    """Three topics; topic 2 appeared recently; one term spans topics 0/1."""
    rng = np.random.default_rng(0)
    concepts = []
    for axis, first_seen, name in (
        (0, OLD, "vision"),
        (1, OLD, "speech"),
        (2, NEW, "agents"),
    ):
        for index in range(5):
            concepts.append(
                TaxonomyConcept(
                    concept_id=f"{name}:{index}",
                    label=f"{name} {index}",
                    kind="Technology",
                    vector=vector(axis, 0.05, rng),
                    first_seen=first_seen,
                    document_dates=[first_seen] * (index + 1),
                )
            )
    general = np.zeros(8)
    general[0] = general[1] = 1.0
    concepts.append(
        TaxonomyConcept(
            concept_id="general:ml",
            label="machine learning",
            kind="Technology",
            vector=list(general),
            first_seen=OLD,
            document_dates=[OLD] * 10,
        )
    )
    concepts.append(
        TaxonomyConcept(
            concept_id="future:1",
            label="future tech",
            kind="Technology",
            vector=vector(3, 0.05, rng),
            first_seen=date(2027, 1, 1),
        )
    )
    return concepts


def config(**overrides):
    value = dict(load_catalog("taxonomy"))
    value.update(overrides)
    return value


def test_topics_become_branches_and_general_terms_stay_at_the_parent():
    taxonomy = build_taxonomy(corpus(), "2026-01-01", config=config())
    root = taxonomy.root()
    assert "future:1" not in taxonomy.concepts, "no concepts from the future"
    branches = [
        {cid.split(":")[0] for cid in taxonomy.nodes[child].subtree_ids}
        for child in root.children
    ]
    assert sorted(map(sorted, branches)) == [
        ["agents"],
        ["speech"],
        ["vision"],
    ]
    assert "general:ml" in root.concept_ids
    assert "general:ml" in taxonomy.general_terms
    assert all(" / " in taxonomy.nodes[c].label for c in root.children)


def test_new_branch_in_known_area_and_semantic_novelty():
    taxonomy = build_taxonomy(corpus(), "2026-01-01", config=config())
    features = taxonomy_features(taxonomy, config())
    agents, vision = features["agents:0"], features["vision:0"]
    assert agents["new_branch_in_known_area"] is True
    assert agents["branch_new_share"] == 1.0
    assert vision["new_branch_in_known_area"] is False
    # An agent term is far from everything known a year earlier; a vision
    # term has an established neighbour.
    assert agents["semantic_novelty"] > 0.5 > vision["semantic_novelty"]
    assert features["general:ml"]["taxonomy_general_term"] is True


def test_explicit_parent_pulls_a_child_into_its_branch():
    concepts = corpus()
    rng = np.random.default_rng(1)
    concepts.append(
        TaxonomyConcept(
            concept_id="child:x",
            label="edge vision chip",
            kind="Technology",
            # Between speech and vision, slightly closer to speech.
            vector=list(
                np.array(vector(1, 0.0, rng)) * 0.55
                + np.array(vector(0, 0.0, rng)) * 0.45
            ),
            first_seen=OLD,
        )
    )
    taxonomy = build_taxonomy(
        concepts,
        "2026-01-01",
        parents=[("child:x", "vision:0")],
        config=config(),
    )
    node = taxonomy.nodes[taxonomy.placement["child:x"]]
    assert "vision:0" in node.subtree_ids


def test_taxonomy_is_deterministic_and_versioned_by_snapshot():
    first = build_taxonomy(corpus(), "2026-01-01", config=config())
    second = build_taxonomy(corpus(), "2026-01-01", config=config())
    assert first.version == second.version
    assert first.placement == second.placement
    earlier = build_taxonomy(corpus(), "2024-01-01", config=config())
    assert earlier.version != first.version
    assert not any(cid.startswith("agents") for cid in earlier.concepts)


def test_feature_rows_carry_taxonomy_columns():
    taxonomy = build_taxonomy(corpus(), "2026-01-01", config=config())
    features = taxonomy_features(taxonomy, config())
    rows = build_feature_rows(
        [
            {
                "technology_id": "agents:0",
                "technology": "agents 0",
                "document_id": "d",
                "mentions": 1,
            }
        ],
        [
            {
                "document_id": "d",
                "created_at": "2025-06-01",
                "source_id": "s",
                "source_family": "scholarly",
                "independence_group": None,
                "metrics_json": "{}",
                "countries": [],
                "companies": [],
                "universities": [],
                "domains": [],
            }
        ],
        [],
        "2026-01-01",
        taxonomy=features,
    )
    assert rows[0]["new_branch_in_known_area"] is True
    assert (
        rows[0]["semantic_novelty"] == features["agents:0"]["semantic_novelty"]
    )


class Result:
    async def consume(self):
        return None


class Transaction:
    def __init__(self):
        self.queries = []

    async def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return Result()


def test_taxonomy_is_written_as_versioned_nodes_and_placements():
    taxonomy = build_taxonomy(corpus(), "2026-01-01", config=config())
    tx = Transaction()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute_write(self, callback, *args):
            return await callback(tx, *args)

    class Driver:
        def session(self, **kwargs):
            return Session()

    store = object.__new__(GraphStore)
    store._driver = Driver()
    store._database = "offline"
    asyncio.run(store.write_taxonomy(taxonomy))
    queries = "\n".join(query for query, _ in tx.queries)
    assert "DETACH DELETE n" in queries
    assert "MERGE (n:TaxonomyNode" in queries
    assert "MERGE (child)-[:CHILD_OF]->(parent)" in queries
    placement = next(
        parameters
        for query, parameters in tx.queries
        if "IN_TAXONOMY" in query
    )
    assert placement["version"] == taxonomy.version
    assert {row["concept_id"] for row in placement["rows"]} == set(
        taxonomy.placement
    )
