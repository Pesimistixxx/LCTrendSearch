import json
import sys
from datetime import date
from types import ModuleType, SimpleNamespace

import pytest

from lctrend.graph.subgraphs import (
    sample_subgraph,
    to_pyg,
    write_subgraph_rows,
)
from lctrend.graph.temporal import TemporalCorpus


def data():
    return {
        "versions": [
            {
                "document_id": "doc",
                "version_id": "v1",
                "document_type": "repository",
                "source_id": "github",
                "version_published_at": "2019-01-01",
                "retrieved_at": "2019-01-02",
                "metrics_observed_at": "2021-01-01",
                "metrics_json": {"stars": 999},
                "companies": ["same"],
                "contributors": ["person"],
                "domains": ["ai"],
            },
            {
                "document_id": "future",
                "version_id": "v2",
                "version_published_at": "2022-01-01",
                "companies": ["future-company"],
            },
        ],
        "mentions": [
            {
                "technology_id": "same",
                "version_id": "v1",
                "observed_at": "2019-01-02",
                "mentions": 3,
            },
            {
                "technology_id": "future-tech",
                "version_id": "v2",
                "observed_at": "2022-01-01",
                "mentions": 100,
            },
        ],
        "relations": [
            {
                "technology_id": "same",
                "version_id": "v1",
                "relation": "USED_BY",
                "target_id": "same",
                "target_kind": "Company",
                "observed_at": "2019-01-02",
            }
        ],
        "technologies": [
            {
                "technology_id": "same",
                "embedding": [1, 0],
                "embedding_observed_at": "2021-01-01",
            }
        ],
    }


def test_sample_is_typed_dated_and_excludes_future_values(tmp_path):
    snapshot = TemporalCorpus(data()).view(date(2020, 1, 1))
    sample = sample_subgraph(
        snapshot, "same", label=1, split="train", config={"hops": 3}
    )
    nodes = {node["id"]: node for node in sample["nodes"]}
    assert "Technology:same" in nodes
    assert "Company:same" in nodes
    assert nodes["DocumentVersion:v1"]["features"]["stars"] is None
    assert nodes["DocumentVersion:v1"]["missing_mask"]["stars"] == 1
    assert not any("future" in node_id for node_id in nodes)
    assert not any(
        key.startswith("embedding_")
        for key in nodes["Technology:same"]["features"]
    )
    assert all(
        node["timestamp"] <= sample["snapshot"] for node in sample["nodes"]
    )
    assert all(
        edge["timestamp"] <= sample["snapshot"] for edge in sample["edges"]
    )
    assert sample["label"] == 1
    assert all("label" not in node["features"] for node in sample["nodes"])
    path = tmp_path / "samples.jsonl"
    assert write_subgraph_rows(path, [sample]) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == sample


def test_sampling_is_bounded_and_stable_when_input_order_changes():
    raw = data()
    first = sample_subgraph(
        TemporalCorpus(raw).view(date(2020, 1, 1)),
        "same",
        config={"hops": 3, "max_neighbors": 1, "max_nodes": 3},
    )
    for values in raw.values():
        values.reverse()
    second = sample_subgraph(
        TemporalCorpus(raw).view(date(2020, 1, 1)),
        "same",
        config={"hops": 3, "max_neighbors": 1, "max_nodes": 3},
    )
    assert second == first
    assert len(first["nodes"]) <= 3
    assert first["root_id"] in {node["id"] for node in first["nodes"]}


def test_label_columns_cannot_be_smuggled_into_graph_features():
    snapshot = TemporalCorpus(data()).view(date(2020, 1, 1))
    with pytest.raises(ValueError, match="Outcome fields"):
        sample_subgraph(
            snapshot, "same", features={"future_document_count": 9}
        )
    with pytest.raises(KeyError, match="not known"):
        sample_subgraph(snapshot, "future-tech")


def test_cached_graph_is_shared_without_mutating_root_features():
    snapshot = TemporalCorpus(data()).view(date(2020, 1, 1))
    enriched = sample_subgraph(snapshot, "same", features={"custom": 7})
    cache = snapshot._typed_subgraph_cache
    plain = sample_subgraph(snapshot, "same")
    assert snapshot._typed_subgraph_cache is cache
    assert (
        next(
            node
            for node in enriched["nodes"]
            if node["id"] == enriched["root_id"]
        )["features"]["custom"]
        == 7
    )
    assert (
        "custom"
        not in next(
            node for node in plain["nodes"] if node["id"] == plain["root_id"]
        )["features"]
    )


def test_manifest_aligns_pyg_columns_for_missing_vectors(
    tmp_path, monkeypatch
):
    corpus = TemporalCorpus(data())
    early = sample_subgraph(corpus.view(date(2020, 1, 1)), "same")
    later = sample_subgraph(corpus.view(date(2021, 1, 1)), "same")
    path = tmp_path / "graphs.jsonl"
    write_subgraph_rows(path, [early, later])
    manifest = json.loads(
        path.with_name(path.name + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["sample_count"] == 2
    assert manifest["node_features"]["Technology"] == [
        "document_count",
        "embedding_0",
        "embedding_1",
    ]

    class Tensor:
        def __init__(self, values, dtype=None):
            self.values, self.dtype = values, dtype

        def reshape(self, *shape):
            self.shape = shape
            return self

    class Graph(dict):
        def __getitem__(self, key):
            return self.setdefault(key, SimpleNamespace())

    torch = ModuleType("torch")
    torch.tensor = Tensor
    torch.float32, torch.bool, torch.long = "float32", "bool", "long"
    pyg = ModuleType("torch_geometric")
    pyg_data = ModuleType("torch_geometric.data")
    pyg_data.HeteroData = Graph
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_geometric", pyg)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", pyg_data)
    first, second = to_pyg(early, manifest), to_pyg(later, manifest)
    assert (
        first["Technology"].x.shape == second["Technology"].x.shape == (1, 3)
    )
    assert first["Technology"].missing_mask.values == [[False, True, True]]
    assert second["Technology"].missing_mask.values == [[False, False, False]]
    assert (
        first["Technology"].feature_names == second["Technology"].feature_names
    )
