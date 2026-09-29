"""Neutral, bounded technology-centred heterogeneous snapshot samples.

Labels and split assignments belong to the sample envelope and never node
features. ``missing_mask`` uses 1 for unknown, 0 for observed; this applies
to node and edge feature dictionaries. PyG is an optional conversion only.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
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


def _prepare(snapshot, technology_id, features):
    """Root id, nodes (root features added) and cached adjacency."""
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
    return root, nodes, adjacency


def _sample(
    snapshot, technology_id, root, nodes, selected, label, split, sampling
):
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
        "sampling": sampling,
        "nodes": [nodes[key] for key in sorted(selected)],
        "edges": retained_edges,
    }


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
    root, nodes, adjacency = _prepare(snapshot, technology_id, features)

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
    return _sample(
        snapshot,
        technology_id,
        root,
        nodes,
        selected,
        label,
        split,
        {
            "hops": hops,
            "max_neighbors": max_neighbors,
            "max_nodes": max_nodes,
            "seed": seed,
        },
    )


# Traversal passes only through technologies and documents. Hubs such as
# countries, companies, domains and tasks are dead ends: in the subgraph,
# never a path to further technologies. Anything else (authors, sources,
# assertions, evidence) is already a feature and stays out.
TECHNOLOGY_TYPES = ("Technology", "Method", "Material")
DOCUMENT_TYPE = "DocumentVersion"
DEFAULT_NEIGHBORHOOD = {
    "leaf_types": [
        "Company",
        "University",
        "Organization",
        "Country",
        "Domain",
        "Task",
    ],
    "max_nodes": 400,
    "max_document_technologies": 30,
    "first_hop": {
        "documents_by_age": {"recent": 12, "middle": 10, "old": 10},
        "technologies": 10,
        "leaves_per_type": 10,
    },
    "second_hop": {
        "technologies_per_document": 5,
        "documents_per_technology": 3,
        "leaves_per_type": 3,
    },
    "seed": 13,
}


def sample_neighborhood(
    snapshot: SnapshotView,
    technology_id: str,
    label: Optional[int] = None,
    split: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Two hops through technologies and documents; hubs are dead ends.

    Hop 1: documents of the root by age (a quota each for the last year,
    one to three years, older; the most recent first within a quota, so an
    empty recent quota itself says the technology fades), related
    technologies, and dead-end hubs. Hop 2: from each document the other
    technologies it mentions and its hubs; from each related technology its
    latest documents and hubs. A document mentioning
    ``max_document_technologies`` or more technologies (a survey) is kept
    but not crossed. Nothing is expanded from a hub or at hop 2. Edges are
    all those among the selected nodes. Order never depends on input order.
    """
    settings = {**DEFAULT_NEIGHBORHOOD, **(config or {})}
    first, second = settings["first_hop"], settings["second_hop"]
    leaf_types = set(settings["leaf_types"])
    max_nodes = int(settings["max_nodes"])
    seed = int(settings["seed"])
    root, nodes, adjacency = _prepare(snapshot, technology_id, features)
    cutoff = snapshot.cutoff

    def kind(identifier):
        return nodes[identifier]["type"]

    def rank(value):
        payload = str(seed) + "|" + root + "|" + value
        return hashlib.sha256(payload.encode()).hexdigest()

    def newest_first(values):
        return sorted(
            values,
            key=lambda value: (nodes[value]["timestamp"], rank(value)),
            reverse=True,
        )

    def neighbors(vertex, types):
        return {
            other
            for others in adjacency[vertex].values()
            for other in others
            if other in nodes and kind(other) in types
        }

    def age_bucket(identifier):
        days = (
            cutoff - date.fromisoformat(nodes[identifier]["timestamp"][:10])
        ).days
        return "recent" if days <= 365 else "middle" if days <= 1095 else "old"

    def is_survey(document):
        mentioned = neighbors(document, set(TECHNOLOGY_TYPES))
        return len(mentioned) >= settings["max_document_technologies"]

    selected = [root]
    chosen = {root}

    def take(values, limit):
        added = []
        for value in values:
            if len(added) >= limit or len(chosen) >= max_nodes:
                break
            if value not in chosen:
                chosen.add(value)
                selected.append(value)
                added.append(value)
        return added

    def take_leaves(vertex, limit):
        by_type = defaultdict(list)
        for value in neighbors(vertex, leaf_types):
            by_type[kind(value)].append(value)
        for name in sorted(by_type):
            take(sorted(by_type[name], key=rank), limit)

    # Hop 1.
    documents = []
    by_age = defaultdict(list)
    for document in neighbors(root, {DOCUMENT_TYPE}):
        by_age[age_bucket(document)].append(document)
    for bucket in ("recent", "middle", "old"):
        documents += take(
            newest_first(by_age[bucket]),
            first["documents_by_age"][bucket],
        )
    related = take(
        sorted(neighbors(root, set(TECHNOLOGY_TYPES)) - {root}, key=rank),
        first["technologies"],
    )
    take_leaves(root, first["leaves_per_type"])
    # Hop 2: only from documents that are not surveys and from technologies.
    for document in documents:
        if is_survey(document):
            continue
        take(
            sorted(neighbors(document, set(TECHNOLOGY_TYPES)), key=rank),
            second["technologies_per_document"],
        )
        take_leaves(document, second["leaves_per_type"])
    for technology in related:
        take(
            newest_first(neighbors(technology, {DOCUMENT_TYPE})),
            second["documents_per_technology"],
        )
        take_leaves(technology, second["leaves_per_type"])
    return _sample(
        snapshot,
        technology_id,
        root,
        nodes,
        set(selected),
        label,
        split,
        {"strategy": "neighborhood", **settings},
    )


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


__all__ = [
    "sample_subgraph",
    "sample_neighborhood",
    "write_subgraph_rows",
    "to_pyg",
]
