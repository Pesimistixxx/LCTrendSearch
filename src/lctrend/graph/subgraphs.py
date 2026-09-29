"""Neutral, bounded technology-centred heterogeneous snapshot samples.

Labels and split assignments belong to the sample envelope and never node
features. ``missing_mask`` uses 1 for unknown, 0 for observed; this applies
to node and edge feature dictionaries. PyG is an optional conversion only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .novelty import _embedding_map, _relations
from .temporal import SnapshotView


def _id(kind, identifier):
    return str(kind) + ":" + str(identifier)


def _masks(features):
    return {key: int(value is None) for key, value in features.items()}


def _numeric(row, names):
    return {
        name: float(row[name])
        if isinstance(row.get(name), (int, float))
        else None
        for name in names
    }


def _snapshot_graph(snapshot):
    cached = getattr(snapshot, "_typed_subgraph_cache", None)
    if cached is not None:
        return cached
    nodes, edges = {}, {}

    def node(kind, identifier, observed, features=None):
        identifier = _id(kind, identifier)
        features = features or {}
        timestamp = observed.isoformat()
        if identifier not in nodes:
            nodes[identifier] = {
                "id": identifier,
                "type": kind,
                "timestamp": timestamp,
                "features": features,
                "missing_mask": _masks(features),
            }
        elif timestamp < nodes[identifier]["timestamp"]:
            nodes[identifier]["timestamp"] = timestamp
        return identifier

    def edge(source, target, kind, observed, features=None):
        features = features or {}
        timestamp = observed.isoformat()
        key = (source, target, kind, timestamp)
        edges[key] = {
            "source": source,
            "target": target,
            "type": kind,
            "timestamp": timestamp,
            "features": features,
            "missing_mask": _masks(features),
        }

    embeddings = _embedding_map(snapshot)
    for key, technology in sorted(snapshot.technologies.items()):
        features = {"document_count": len(technology.documents)}
        embedding = embeddings.get(key)
        # Vector slots stay numeric and carry the same missing mask convention.
        if embedding is not None and len(embedding):
            features.update(
                {
                    "embedding_" + str(i): float(value)
                    for i, value in enumerate(embedding)
                }
            )
        # A technology seen only in undated documents has no first_seen.
        technology_node = node(
            "Technology",
            key,
            technology.first_seen
            or min(trace.first_visible for trace in technology.documents),
            features,
        )
        for trace in technology.documents:
            version = trace.version
            published = version.available_date
            document_node = node(
                "Document", version.document_id, trace.first_visible
            )
            metrics = _numeric(
                trace.metrics or {},
                ("citation_count", "stars", "forks", "downloads"),
            )
            metrics.update(
                {
                    "reliability_tier": version.reliability_tier,
                    "fulltext_available": int(
                        version.coverage in ("full_text", "selected_files")
                    ),
                }
            )
            version_node = node(
                "DocumentVersion", version.version_id, published, metrics
            )
            # Family remains a type-level observable; no current metrics enter.
            nodes[version_node]["source_family"] = version.family
            nodes[version_node]["document_type"] = version.document_type
            # Human-readable provenance is audit metadata, never tensor input.
            reference = snapshot.corpus.document_info.get(
                version.document_id, {}
            )
            nodes[version_node]["title"] = reference.get("title")
            nodes[version_node]["url"] = reference.get("url")
            edge(document_node, version_node, "HAS_VERSION", published)
            edge(
                technology_node,
                version_node,
                "MENTIONED_IN",
                trace.first_visible,
                {"mentions": trace.mention_count},
            )
            if version.source_id:
                source = node("Source", version.source_id, published)
                edge(version_node, source, "FROM_SOURCE", published)
            for kind, values, relation in (
                ("Company", version.companies, "ASSOCIATED_WITH"),
                ("University", version.universities, "ASSOCIATED_WITH"),
                ("Organization", version.organizations, "ASSOCIATED_WITH"),
                ("Country", version.countries, "IN_COUNTRY"),
                ("Domain", version.domains, "IN_DOMAIN"),
                ("Author", version.contributors, "CONTRIBUTED_BY"),
            ):
                for value in sorted(set(values)):
                    party = node(kind, value, published)
                    edge(version_node, party, relation, published)
    for technology_id, event in _relations(snapshot):
        data = event.data
        relation, target_id = data.get("relation"), data.get("target_id")
        if not relation or not target_id:
            continue
        source = node(
            "Technology",
            technology_id,
            event.observed,
            {"document_count": None},
        )
        kind = data.get("target_kind")
        if not kind:
            labels = sorted(set(data.get("target_labels") or []))
            kind = next(
                (
                    label
                    for label in (
                        "Technology",
                        "Task",
                        "Company",
                        "University",
                        "Organization",
                        "Country",
                        "Domain",
                    )
                    if label in labels
                ),
                None,
            )
        kind = kind or {
            "SOLVES": "Task",
            "SUBTECHNOLOGY_OF": "Technology",
            "DEVELOPED_IN": "Country",
        }.get(relation, "Organization")
        target = node(
            str(kind),
            target_id,
            event.observed,
            {"document_count": None} if kind == "Technology" else {},
        )
        edge(source, target, str(relation), event.observed)
    for key, technology in sorted(snapshot.technologies.items()):
        source = _id("Technology", key)
        for kind, events, relation, numeric in (
            (
                "MaturityEvidence",
                technology.maturity,
                "HAS_MATURITY_EVIDENCE",
                ("stage_rank", "trl"),
            ),
            (
                "EconomicEvidence",
                technology.economics,
                "HAS_ECONOMIC_EVIDENCE",
                ("amount_value", "confidence", "reliability_tier"),
            ),
            (
                "Assertion",
                technology.assertions,
                "HAS_ASSERTION",
                ("confidence",),
            ),
        ):
            for event in events:
                data = event.data
                # Unavailable source versions make evidence ineligible.
                version_id = data.get("version_id")
                if not snapshot.corpus.event_visible(event, snapshot.cutoff):
                    continue
                raw_id = data.get("assertion_id") or data.get("evidence_id")
                if raw_id is None:
                    encoded = json.dumps(data, sort_keys=True, default=str)
                    raw_id = hashlib.sha256(encoded.encode()).hexdigest()[:24]
                evidence = node(
                    kind, raw_id, event.observed, _numeric(data, numeric)
                )
                edge(source, evidence, relation, event.observed)
                version_node = _id("DocumentVersion", version_id)
                if version_node in nodes:
                    edge(
                        evidence, version_node, "SUPPORTED_BY", event.observed
                    )
    result = nodes, list(edges.values())
    snapshot._typed_subgraph_cache = result
    return result


def sample_subgraph(
    snapshot: SnapshotView,
    technology_id: str,
    label: Optional[int] = None,
    split: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic BFS, capped per relation/vertex, with a global node cap.

    Only numeric caller-provided root features are permitted. Future outcome
    fields are rejected to keep the label separate from the input graph.
    Traversal treats edges as undirected, while exported edges preserve their
    direction and type. Sampling does not depend on input order.
    """
    config = config or {}
    hops = max(0, int(config.get("hops", 2)))
    max_neighbors = max(0, int(config.get("max_neighbors", 30)))
    max_nodes = max(1, int(config.get("max_nodes", 400)))
    seed = int(config.get("seed", 13))
    root = _id("Technology", technology_id)
    nodes, edges = _snapshot_graph(snapshot)
    if root not in nodes:
        raise KeyError(
            "Technology is not known at snapshot: " + str(technology_id)
        )
    if features:
        # Features should normally be computed by the training builder itself.
        # Silently dropping outcome fields would hide a caller's leakage bug.
        forbidden = [
            name
            for name in features
            if name.startswith(("future_", "label_")) or name == "split"
        ]
        if forbidden:
            raise ValueError(
                "Outcome fields cannot be graph features: "
                + ", ".join(forbidden)
            )
        numeric = {
            name: value
            for name, value in features.items()
            if value is None or isinstance(value, (int, float, bool))
        }
        # Keep the shared snapshot cache immutable across different roots.
        nodes = dict(nodes)
        nodes[root] = dict(nodes[root])
        nodes[root]["features"] = {**nodes[root]["features"], **numeric}
        nodes[root]["missing_mask"] = _masks(nodes[root]["features"])
    adjacency = getattr(snapshot, "_typed_subgraph_adjacency_cache", None)
    if adjacency is None:
        adjacency = defaultdict(lambda: defaultdict(set))
        outgoing = defaultdict(list)
        for edge in edges:
            adjacency[edge["source"]][edge["type"]].add(edge["target"])
            adjacency[edge["target"]][edge["type"]].add(edge["source"])
            outgoing[edge["source"]].append(edge)
        snapshot._typed_subgraph_adjacency_cache = adjacency
        snapshot._typed_subgraph_outgoing_cache = outgoing

    def rank(value):
        payload = str(seed) + "|" + root + "|" + value
        return hashlib.sha256(payload.encode()).hexdigest(), value

    selected, frontier = {root}, [root]
    for _ in range(hops):
        following = set()
        for vertex in sorted(frontier):
            for relation in sorted(adjacency[vertex]):
                for neighbor in sorted(adjacency[vertex][relation], key=rank)[
                    :max_neighbors
                ]:
                    if neighbor not in selected and len(selected) < max_nodes:
                        selected.add(neighbor)
                        following.add(neighbor)
        frontier = sorted(following)
        if not frontier or len(selected) >= max_nodes:
            break
    retained_edges = [
        edge
        for source in selected
        for edge in snapshot._typed_subgraph_outgoing_cache.get(source, ())
        if edge["target"] in selected
    ]
    retained_edges.sort(
        key=lambda edge: (
            edge["source"],
            edge["type"],
            edge["target"],
            edge["timestamp"],
        )
    )
    return {
        "schema_version": 1,
        "technology_id": str(technology_id),
        "root_id": root,
        "snapshot": snapshot.cutoff.isoformat(),
        "label": label,
        "split": split,
        "mask_semantics": "1=missing, 0=observed",
        "sampling": {
            "hops": hops,
            "max_neighbors": max_neighbors,
            "max_nodes": max_nodes,
            "seed": seed,
        },
        "nodes": [nodes[key] for key in sorted(selected)],
        "edges": retained_edges,
    }


def _edge_schema_key(source_type, relation, target_type):
    return "|".join((source_type, relation, target_type))


def write_subgraph_rows(path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write JSONL and ``<path>.manifest.json`` with dataset-wide columns.

    Pass the resulting manifest as ``feature_schema`` to :func:`to_pyg` so
    every sample uses the same feature width, including absent embeddings.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    node_features, edge_features = defaultdict(set), defaultdict(set)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            types = {}
            for node in row["nodes"]:
                types[node["id"]] = node["type"]
                node_features[node["type"]].update(node["features"])
            for edge in row["edges"]:
                key = _edge_schema_key(
                    types[edge["source"]], edge["type"], types[edge["target"]]
                )
                edge_features[key].update(edge["features"])
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, allow_nan=False, sort_keys=True
                )
                + "\n"
            )
            count += 1
    manifest = {
        "schema_version": 1,
        "format": "lctrend-typed-subgraphs-jsonl",
        "sample_count": count,
        "data_file": path.name,
        "mask_semantics": "1=missing, 0=observed",
        "missing_value_fill": 0,
        "node_features": {
            key: sorted(names) for key, names in sorted(node_features.items())
        },
        "edge_features": {
            key: sorted(names) for key, names in sorted(edge_features.items())
        },
        "edge_type_separator": "|",
        "label_location": "sample.label",
    }
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return count


def to_pyg(sample, feature_schema=None):
    """Convert JSON sample to PyG HeteroData (requires torch + PyG).

    Numeric feature columns are sorted and stored as ``feature_names`` on
    each type; absent values become zero with a matching ``missing_mask``.
    ``feature_schema`` is the manifest written by :func:`write_subgraph_rows`;
    use it when converting multiple samples to keep dataset tensors aligned.
    Directed edge types, indices, node IDs, timestamps and root are retained.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError(
            "PyG serialization requires torch and torch-geometric"
        ) from exc
    result = HeteroData()
    feature_schema = feature_schema or {}
    typed = defaultdict(list)
    for node in sample["nodes"]:
        typed[node["type"]].append(node)
    indices = {}

    def tensors(records, names=None):
        names = (
            sorted({name for record in records for name in record["features"]})
            if names is None
            else list(names)
        )
        values, masks = [], []
        for record in records:
            values.append(
                [float(record["features"].get(name) or 0) for name in names]
            )
            masks.append(
                [record["features"].get(name) is None for name in names]
            )
        return (
            names,
            torch.tensor(values, dtype=torch.float32).reshape(
                len(records), len(names)
            ),
            torch.tensor(masks, dtype=torch.bool).reshape(
                len(records), len(names)
            ),
        )

    for kind in feature_schema.get("node_features", {}):
        typed.setdefault(kind, [])
    for kind, records in sorted(typed.items()):
        names, values, masks = tensors(
            records, feature_schema.get("node_features", {}).get(kind)
        )
        result[kind].num_nodes = len(records)
        result[kind].x = values
        result[kind].missing_mask = masks
        result[kind].feature_names = names
        result[kind].node_ids = [record["id"] for record in records]
        result[kind].timestamps = [record["timestamp"] for record in records]
        indices.update(
            {
                record["id"]: (kind, index)
                for index, record in enumerate(records)
            }
        )
    edge_types = defaultdict(list)
    for edge in sample["edges"]:
        left, right = indices[edge["source"]], indices[edge["target"]]
        edge_types[(left[0], edge["type"], right[0])].append(edge)
    for kind in feature_schema.get("edge_features", {}):
        parts = tuple(kind.split("|"))
        if len(parts) != 3:
            raise ValueError("Invalid edge type in feature schema: " + kind)
        edge_types.setdefault(parts, [])
    for kind, records in sorted(edge_types.items()):
        result[kind].edge_index = torch.tensor(
            [
                [indices[record["source"]][1] for record in records],
                [indices[record["target"]][1] for record in records],
            ],
            dtype=torch.long,
        )
        schema_key = _edge_schema_key(*kind)
        names, values, masks = tensors(
            records, feature_schema.get("edge_features", {}).get(schema_key)
        )
        result[kind].edge_attr = values
        result[kind].missing_mask = masks
        result[kind].feature_names = names
        result[kind].timestamps = [record["timestamp"] for record in records]
    result.snapshot = sample["snapshot"]
    result.root_type, result.root_index = indices[sample["root_id"]]
    result.label_missing = sample["label"] is None
    result.y = torch.tensor([sample["label"] or 0], dtype=torch.long)
    result.split = sample["split"]
    return result


__all__ = ["sample_subgraph", "write_subgraph_rows", "to_pyg"]
