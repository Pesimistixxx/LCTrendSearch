"""Graph and semantic context recomputed from a dated snapshot.

Technology topology is the undirected projection of co-mentions and dated
technology-to-technology relations. It deliberately does not read the live
Neo4j projection or stored taxonomy. All graph algorithms are deterministic;
Brandes betweenness can sample source nodes to bound its cost.
"""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from datetime import timedelta
from itertools import combinations
from typing import Any, Dict, Optional

import numpy as np

from .features import entropy, months_before
from .temporal import SnapshotView

NOVELTY_FIELDS = (
    "taxonomy_level",
    "taxonomy_node_size",
    "taxonomy_sibling_count",
    "taxonomy_general_term",
    "taxonomy_depth",
    "semantic_novelty",
    "nearest_known_distance",
    "cluster_centroid_distance",
    "embedding_outlier_score",
    "branch_growth",
    "branch_new_share",
    "new_branch_in_known_area",
    "degree",
    "degree_delta_12m",
    "pagerank",
    "pagerank_delta_12m",
    "betweenness",
    "betweenness_delta_12m",
    "k_core",
    "k_core_delta_12m",
    "community_crossing_count",
    "neighbor_domain_entropy",
    "structural_hole_score",
    "new_domain_pair_count",
    "bridge_score",
    "recombination_surprise",
    "task_combination_novelty",
    "nearest_known_technology_distance",
    "semantic_outlier_score",
    "new_taxonomy_branch",
)


def _relations(snapshot: SnapshotView):
    """Include dated relation-only concepts, retaining version availability."""
    for source, kinds in sorted(snapshot.corpus.events.items()):
        for event in kinds.get("relations", []):
            if hasattr(snapshot.corpus, "event_visible"):
                if snapshot.corpus.event_visible(event, snapshot.cutoff):
                    yield source, event
                continue
            version_id = event.data.get("version_id")
            version = snapshot.corpus.versions.get(version_id)
            if event.observed > snapshot.cutoff:
                continue
            if version_id and (
                version is None or version.available_date > snapshot.cutoff
            ):
                continue
            yield source, event


def technology_graph(snapshot: SnapshotView) -> Dict[str, set]:
    """Snapshot technology adjacency; shared documents create one edge."""
    graph = {key: set() for key in sorted(snapshot.technologies)}
    documents: Dict[str, set] = {}
    for key, technology in snapshot.technologies.items():
        for trace in technology.documents:
            documents.setdefault(trace.document_id, set()).add(key)
    for ids in documents.values():
        for left, right in combinations(sorted(ids), 2):
            graph[left].add(right)
            graph[right].add(left)
    for source, event in _relations(snapshot):
        data = event.data
        target = data.get("target_id")
        technology = (
            data.get("relation") == "SUBTECHNOLOGY_OF"
            or data.get("target_kind") == "Technology"
            or "Technology" in (data.get("target_labels") or [])
        )
        if technology and target and source != str(target):
            target = str(target)
            graph.setdefault(source, set()).add(target)
            graph.setdefault(target, set()).add(source)
    return graph


def _pagerank(graph: Dict[str, set]) -> Dict[str, float]:
    if not graph:
        return {}
    size, damping = len(graph), 0.85
    ranks = dict.fromkeys(graph, 1.0 / size)
    for _ in range(100):
        dangling = sum(ranks[key] for key in graph if not graph[key])
        updated = dict.fromkeys(
            graph, (1.0 - damping + damping * dangling) / size
        )
        for source, neighbors in graph.items():
            for target in sorted(neighbors):
                updated[target] += damping * ranks[source] / len(neighbors)
        change = sum(abs(updated[key] - ranks[key]) for key in graph)
        ranks = updated
        if change < 1e-10:
            break
    return ranks


def _betweenness(graph: Dict[str, set], samples: int, seed: int):
    ids = sorted(graph)
    sources = (
        sorted(random.Random(seed).sample(ids, samples))
        if 0 < samples < len(ids)
        else ids
    )
    values = dict.fromkeys(ids, 0.0)
    for source in sources:
        stack, queue = [], deque([source])
        predecessors = {key: [] for key in ids}
        paths = dict.fromkeys(ids, 0.0)
        paths[source] = 1.0
        distances = {source: 0}
        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbor in sorted(graph[vertex]):
                if neighbor not in distances:
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
                if distances[neighbor] == distances[vertex] + 1:
                    paths[neighbor] += paths[vertex]
                    predecessors[neighbor].append(vertex)
        dependencies = dict.fromkeys(ids, 0.0)
        for vertex in reversed(stack):
            for predecessor in predecessors[vertex]:
                dependencies[predecessor] += (
                    paths[predecessor]
                    / paths[vertex]
                    * (1.0 + dependencies[vertex])
                )
            if vertex != source:
                values[vertex] += dependencies[vertex]
    size = len(ids)
    scale = (
        size / len(sources) / ((size - 1) * (size - 2))
        if size > 2 and sources
        else 0.0
    )
    return {key: value * scale for key, value in values.items()}


def _cores(graph: Dict[str, set]):
    remaining = {key: len(value) for key, value in graph.items()}
    cores, level = {}, 0
    while remaining:
        node = min(remaining, key=lambda key: (remaining[key], key))
        level = max(level, remaining.pop(node))
        cores[node] = level
        for neighbor in graph[node]:
            if neighbor in remaining:
                remaining[neighbor] -= 1
    return cores


def _communities(graph: Dict[str, set]):
    """Deterministic asynchronous label propagation (ties retain label)."""
    labels = {key: key for key in graph}
    for _ in range(100):
        changed = False
        for key in sorted(graph):
            counts = Counter(labels[neighbor] for neighbor in graph[key])
            if not counts:
                continue
            maximum = max(counts.values())
            choices = sorted(
                label for label, n in counts.items() if n == maximum
            )
            label = labels[key] if labels[key] in choices else choices[0]
            if label != labels[key]:
                labels[key] = label
                changed = True
        if not changed:
            break
    return labels


def _graph_metrics(graph, config):
    rank = _pagerank(graph)
    between = _betweenness(
        graph,
        int(config.get("betweenness_samples", 256)),
        int(config.get("seed", 13)),
    )
    cores = _cores(graph)
    return {
        key: {
            "degree": len(graph[key]),
            "pagerank": rank[key],
            "betweenness": between[key],
            "k_core": cores[key],
        }
        for key in graph
    }


def _domains(snapshot):
    return {
        key: {
            domain
            for trace in technology.documents
            for domain in trace.version.domains
        }
        for key, technology in snapshot.technologies.items()
    }


def _tasks(snapshot):
    tasks: Dict[str, set] = {}
    for source, event in _relations(snapshot):
        if event.data.get("relation") == "SOLVES" and event.data.get(
            "target_id"
        ):
            tasks.setdefault(source, set()).add(str(event.data["target_id"]))
    return tasks


def _structural_hole(graph, key):
    """One minus Burt's unweighted network constraint; isolate is missing."""
    neighbors = graph.get(key, set())
    if not neighbors:
        return None
    weight = 1.0 / len(neighbors)
    constraint = 0.0
    for target in neighbors:
        indirect = sum(
            weight / len(graph[other])
            for other in neighbors
            if target in graph[other] and graph[other]
        )
        constraint += (weight + indirect) ** 2
    return max(0.0, min(1.0, 1.0 - constraint))


def _embedding_map(snapshot):
    corpus = snapshot.corpus
    if hasattr(corpus, "embeddings_at"):
        return corpus.embeddings_at(snapshot.cutoff)
    # Compatibility for in-memory corpora without a dated embedding API.
    # Such corpora must not expose undated live vectors as historical facts.
    return {}


def _semantic(snapshot, config):
    from ..taxonomy.builder import (
        TaxonomyConcept,
        build_taxonomy,
        taxonomy_features,
    )

    embeddings = _embedding_map(snapshot)
    vectors = {}
    dimension = None
    for key in sorted(snapshot.technologies):
        raw = embeddings.get(key)
        if raw is None:
            continue
        vector = np.asarray(raw, dtype=float)
        if (
            vector.ndim != 1
            or not len(vector)
            or not np.isfinite(vector).all()
        ):
            continue
        norm = float(np.linalg.norm(vector))
        if not norm or (dimension is not None and len(vector) != dimension):
            continue
        dimension = len(vector)
        vectors[key] = vector / norm
    concepts = [
        TaxonomyConcept(
            key,
            snapshot.technologies[key].label,
            "Technology",
            vectors[key],
            snapshot.technologies[key].first_seen,
            [
                trace.first_visible
                for trace in snapshot.technologies[key].documents
            ],
        )
        for key in vectors
    ]
    parents = [
        (source, str(event.data["target_id"]))
        for source, event in _relations(snapshot)
        if event.data.get("relation") == "SUBTECHNOLOGY_OF"
        and event.data.get("target_id")
    ]
    taxonomy = build_taxonomy(concepts, snapshot.cutoff.isoformat(), parents)
    rows = taxonomy_features(taxonomy) if concepts else {}
    neighbors = max(1, int(config.get("outlier_neighbors", 5)))
    for key in snapshot.technologies:
        row = rows.setdefault(key, {})
        vector = vectors.get(key)
        references = [
            other
            for other in vectors
            if other != key
            and snapshot.technologies[other].first_seen < snapshot.cutoff
        ]
        distances = (
            sorted(
                max(0.0, min(2.0, 1.0 - float(vectors[other] @ vector)))
                for other in references
            )
            if vector is not None
            else []
        )
        nearest = distances[0] if distances else None
        row["nearest_known_distance"] = nearest
        row["nearest_known_technology_distance"] = nearest
        row["semantic_novelty"] = nearest
        row["embedding_outlier_score"] = (
            sum(distances[:neighbors]) / len(distances[:neighbors])
            if distances
            else None
        )
        row["semantic_outlier_score"] = row["embedding_outlier_score"]
        node = taxonomy.nodes.get(taxonomy.placement.get(key))
        centroid = None
        if node is not None and vector is not None:
            peers = [
                vectors[other]
                for other in node.subtree_ids
                if other != key and other in references
            ]
            if peers:
                mean = np.sum(peers, axis=0)
                norm = np.linalg.norm(mean)
                if norm:
                    centroid = max(
                        0.0, min(2.0, 1.0 - float(vector @ (mean / norm)))
                    )
        row["cluster_centroid_distance"] = centroid
        row["taxonomy_depth"] = node.level if node else None
        for name in (
            "taxonomy_level",
            "taxonomy_node_size",
            "taxonomy_sibling_count",
            "taxonomy_general_term",
            "branch_growth",
            "branch_new_share",
            "new_branch_in_known_area",
        ):
            row.setdefault(name, None)
        row["new_taxonomy_branch"] = row["new_branch_in_known_area"]
    return rows


def novelty_features(
    snapshot: SnapshotView,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute shared context once, returning one feature dict per technology.

    ``config`` accepts the dataset catalog (``graph``/``semantic`` sections).
    Delta columns subtract the same metric at the calendar date 12 months
    earlier; absent earlier technologies have a zero topology baseline.
    Semantic references exclude the candidate and concepts first seen at T.
    Missing vectors/domain/task evidence produce None, never invented zeros.
    """
    config = config or {}
    graph_config = config.get("graph", config)
    semantic_config = config.get("semantic", {})
    graph = technology_graph(snapshot)
    prior = snapshot.corpus.view(months_before(snapshot.cutoff, 12))
    earlier_graph = technology_graph(prior)
    current_metrics = _graph_metrics(graph, graph_config)
    earlier_metrics = _graph_metrics(earlier_graph, graph_config)
    communities = _communities(graph)
    domains, previous_domains = _domains(snapshot), _domains(prior)
    pairs = Counter(
        pair
        for values in previous_domains.values()
        for pair in combinations(sorted(values), 2)
    )
    tasks = _tasks(snapshot)
    strict_prior = snapshot.corpus.view(snapshot.cutoff - timedelta(days=1))
    prior_tasks = _tasks(strict_prior)
    rows = _semantic(snapshot, semantic_config)
    for key in sorted(snapshot.technologies):
        row = rows[key]
        for metric, value in current_metrics[key].items():
            row[metric] = value
            row[metric + "_delta_12m"] = value - earlier_metrics.get(
                key, {}
            ).get(metric, 0)
        neighbors = graph[key]
        counts = Counter(
            domain for other in neighbors for domain in domains.get(other, ())
        )
        row["community_crossing_count"] = len(
            {
                communities[other]
                for other in neighbors
                if communities[other] != communities[key]
            }
        )
        row["neighbor_domain_entropy"] = entropy(counts.values())
        hole = _structural_hole(graph, key)
        row["structural_hole_score"] = hole
        current_pairs = list(combinations(sorted(domains.get(key, ())), 2))
        row["new_domain_pair_count"] = (
            sum(pair not in pairs for pair in current_pairs)
            if domains.get(key)
            else None
        )
        row["bridge_score"] = (
            hole * (entropy(counts.values()) or 0.0)
            if hole is not None and counts
            else None
        )
        denominator = (
            sum(pairs.values()) + len(set(pairs) | set(current_pairs)) + 1
        )
        row["recombination_surprise"] = (
            sum(
                -math.log((pairs[pair] + 1) / denominator)
                for pair in current_pairs
            )
            / len(current_pairs)
            if current_pairs
            else None
        )
        own_tasks = tasks.get(key, set())
        comparisons = (
            [
                1.0 - len(own_tasks & values) / len(own_tasks | values)
                for other, values in prior_tasks.items()
                if other != key and values
            ]
            if own_tasks
            else []
        )
        row["task_combination_novelty"] = (
            min(comparisons) if comparisons else None
        )
    return rows


__all__ = ["NOVELTY_FIELDS", "novelty_features", "technology_graph"]
