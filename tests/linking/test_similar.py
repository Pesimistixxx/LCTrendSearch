"""SIMILAR_TO: mutual nearest neighbours, kept out of structural metrics."""

import asyncio
from datetime import date

import numpy as np

from lctrend.graph.novelty import technology_graph
from lctrend.linking.similar import METHOD, mutual_neighbors, rebuild


def unit(*values):
    vector = np.asarray(values, dtype=float)
    return list(vector / np.linalg.norm(vector))


def test_only_mutual_neighbours_of_one_family_above_the_floor():
    ids = ["hub", "a", "b", "c", "method", "company"]
    vectors = [
        unit(1, 1, 1),  # an umbrella name near everything
        unit(1, 0.1, 0),
        unit(1, 0.15, 0),
        unit(0, 1, 0.1),
        unit(1, 0.12, 0.01),
        unit(1, 0.11, 0),
    ]
    families = [
        "technology",
        "technology",
        "technology",
        "technology",
        "technology",
        "organization",
    ]

    edges = mutual_neighbors(ids, vectors, families, k=2, min_cosine=0.7)

    pairs = {(left, right) for left, right, _ in edges}
    # a, b and the method are each other's nearest; the hub is in their
    # top-2 lists only one way; the company is of another family.
    assert pairs == {("a", "b"), ("a", "method"), ("b", "method")}
    assert all(cosine >= 0.7 for _, _, cosine in edges)


def test_small_inputs_give_no_edges():
    assert mutual_neighbors(["a"], [unit(1, 0)], ["technology"], 3, 0) == []
    assert mutual_neighbors([], [], [], 3, 0) == []


class Store:
    """Records the statements of a rebuild; returns the given vectors."""

    def __init__(self, rows):
        self.rows = rows
        self.writes = []
        self._database = None
        self._driver = self

    def session(self, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def run(self, query, **parameters):
        if "RETURN c.concept_id" in query:
            return self.rows
        self.writes.append((query, parameters))
        return self

    def consume(self):
        return None

    async def execute_write(self, callback):
        return await callback(self)


def test_rebuild_replaces_the_layer_with_dated_edges():
    rows = [
        {
            "concept_id": "a",
            "kind": "Technology",
            "vector": unit(1, 0.1),
            "first_seen_at": "2021-03-01",
        },
        {
            "concept_id": "b",
            "kind": "Method",
            "vector": unit(1, 0.12),
            "first_seen_at": "2023-05-02",
        },
    ]
    store = Store(rows)

    summary = asyncio.run(rebuild(store, "model-x", k=1, min_cosine=0.5))

    assert summary["edges"] == 1 and summary["written"]
    delete, write = store.writes
    assert "DELETE r" in delete[0]
    assert delete[1] == {"method": METHOD, "model": "model-x"}
    assert ":Technology" in write[0] and ":Method" in write[0]
    [edge] = write[1]["rows"]
    # The edge exists from the day its later concept appeared.
    assert edge["observed_at"] == "2023-05-02"
    assert (edge["left"], edge["right"]) == ("a", "b")


def test_dry_run_writes_nothing():
    store = Store([])
    summary = asyncio.run(rebuild(store, "model-x", dry_run=True))

    assert summary["edges"] == 0 and not summary["written"]
    assert store.writes == []


class Event:
    def __init__(self, data):
        self.data = data
        self.observed = date(2020, 1, 1)


class Corpus:
    def __init__(self, events):
        self.events = events
        self.versions = {}


class Snapshot:
    def __init__(self, events):
        self.technologies = {}
        self.corpus = Corpus(events)
        self.cutoff = date(2024, 1, 1)


def test_similarity_never_becomes_a_structural_edge():
    events = {
        "a": {
            "relations": [
                Event(
                    {
                        "relation": "SIMILAR_TO",
                        "target_id": "b",
                        "target_kind": "Technology",
                    }
                ),
                Event({"relation": "SUBTECHNOLOGY_OF", "target_id": "c"}),
            ]
        }
    }

    graph = technology_graph(Snapshot(events))

    assert graph == {"a": {"c"}, "c": {"a"}}
