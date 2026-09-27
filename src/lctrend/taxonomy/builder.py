"""TaxoGen-style taxonomy for one snapshot date.

TaxoGen (Zhang et al., KDD 2018) splits a term set top-down with spherical
clustering of term embeddings and keeps "general" terms, which are not
concentrated in any one child, at the parent. This is the same recursion over
the stored concept label vectors (``c.embedding``), without TaxoGen's local
embedding retraining. Reviewed ``SUBTECHNOLOGY_OF`` edges pull a child's
vector toward its parent so that an explicit hierarchy from source text
survives clustering.

A taxonomy is built only from concepts known at the snapshot date, so its
features can be used for training without seeing the future.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import load_catalog
from ..core.models import json_value, stable_id


def _date(value: object) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


@dataclass
class TaxonomyConcept:
    """One concept with its label vector and dated document evidence."""

    concept_id: str
    label: str
    kind: str
    vector: Sequence[float]
    first_seen: Optional[date] = None
    document_dates: List[date] = field(default_factory=list)
    document_evidence: List[Tuple[str, date]] = field(default_factory=list)
    embedding_model: Optional[str] = None
    embedding_observed_at: Optional[date] = None

    @classmethod
    def from_row(cls, row: Dict[str, object]) -> "TaxonomyConcept":
        dates = [_date(value) for value in row.get("document_dates") or []]
        return cls(
            concept_id=str(row["concept_id"]),
            label=str(row["label"]),
            kind=str(row["kind"]),
            vector=list(row["embedding"]),
            first_seen=_date(row.get("first_seen_at")),
            document_dates=[value for value in dates if value],
            document_evidence=[
                (str(item["document_id"]), _date(item["date"]))
                for item in row.get("document_evidence") or []
                if item.get("document_id") and _date(item.get("date"))
            ],
            embedding_model=row.get("embedding_model"),
            embedding_observed_at=_date(row.get("embedding_observed_at")),
        )


@dataclass
class TaxonomyNode:
    node_id: str
    path: Tuple[int, ...]
    level: int
    parent_id: Optional[str]
    label: str
    centroid: np.ndarray
    # Concepts attached here: general terms of an inner node or all
    # members of a leaf.
    concept_ids: List[str] = field(default_factory=list)
    # Every concept in the subtree.
    subtree_ids: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    documents_last_year: int = 0
    documents_previous_year: int = 0
    new_share: float = 0.0


@dataclass
class Taxonomy:
    version: str
    snapshot: date
    nodes: Dict[str, TaxonomyNode]
    # concept_id -> node the concept is attached to
    placement: Dict[str, str]
    general_terms: set
    concepts: Dict[str, TaxonomyConcept]

    def root(self) -> TaxonomyNode:
        return next(node for node in self.nodes.values() if node.level == 0)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _spherical_kmeans(
    vectors: np.ndarray, k: int, iterations: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """k-means on the unit sphere (cosine), k-means++ initialization."""
    rng = np.random.default_rng(seed)
    centroids = [vectors[rng.integers(len(vectors))]]
    for _ in range(1, k):
        similarity = np.max(vectors @ np.array(centroids).T, axis=1)
        distance = np.clip(1.0 - similarity, 0.0, None) ** 2
        total = distance.sum()
        if total == 0:
            break
        centroids.append(vectors[rng.choice(len(vectors), p=distance / total)])
    centroids = np.array(centroids)
    assignment = np.zeros(len(vectors), dtype=int)
    for _ in range(iterations):
        new_assignment = np.argmax(vectors @ centroids.T, axis=1)
        updated = []
        for index in range(len(centroids)):
            members = vectors[new_assignment == index]
            updated.append(
                members.sum(axis=0) if len(members) else centroids[index]
            )
        centroids = _unit(np.array(updated))
        if np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
    return assignment, centroids


def _popularity(concept: TaxonomyConcept) -> float:
    count = (
        len({item[0] for item in concept.document_evidence})
        if concept.document_evidence
        else len(concept.document_dates)
    )
    return math.log1p(count)


def _label(
    ids: Sequence[str],
    concepts: Dict[str, TaxonomyConcept],
    vectors: Dict[str, np.ndarray],
    centroid: np.ndarray,
    terms: int,
) -> str:
    """Representative terms: popular and close to the node centroid."""
    ranked = sorted(
        ids,
        key=lambda cid: (
            -(1.0 + _popularity(concepts[cid]))
            * float(vectors[cid] @ centroid),
            concepts[cid].label,
        ),
    )
    return " / ".join(concepts[cid].label for cid in ranked[:terms])


def build_taxonomy(
    concepts: Iterable[TaxonomyConcept],
    snapshot: str,
    parents: Iterable[Tuple[str, str]] = (),
    config: Optional[Dict[str, object]] = None,
) -> Taxonomy:
    """Build the taxonomy of concepts known at ``snapshot``.

    ``parents`` are reviewed (child, parent) SUBTECHNOLOGY_OF pairs.
    """
    config = config or load_catalog("taxonomy")
    cutoff = date.fromisoformat(snapshot[:10])
    kinds = set(config["kinds"])
    known: Dict[str, TaxonomyConcept] = {}
    for concept in concepts:
        dates = [value for value in concept.document_dates if value <= cutoff]
        seen = concept.first_seen or (min(dates) if dates else None)
        if (
            concept.kind not in kinds
            or not seen
            or seen > cutoff
            or (
                concept.embedding_observed_at
                and concept.embedding_observed_at > cutoff
            )
        ):
            continue
        known[concept.concept_id] = TaxonomyConcept(
            concept_id=concept.concept_id,
            label=concept.label,
            kind=concept.kind,
            vector=concept.vector,
            first_seen=seen,
            document_dates=dates,
            document_evidence=[
                item for item in concept.document_evidence if item[1] <= cutoff
            ],
            embedding_model=concept.embedding_model,
            embedding_observed_at=concept.embedding_observed_at,
        )
    parents = sorted(
        set(
            (child, parent)
            for child, parent in parents
            if child in known and parent in known and child != parent
        )
    )
    dimensions = set()
    models = set()
    for concept in known.values():
        vector = np.asarray(concept.vector, dtype=float)
        if (
            vector.ndim != 1
            or not vector.size
            or not np.isfinite(vector).all()
            or not np.linalg.norm(vector)
        ):
            raise ValueError(
                f"Invalid taxonomy embedding for {concept.concept_id}"
            )
        dimensions.add(vector.size)
        models.add(concept.embedding_model)
    if len(dimensions) > 1 or len(models) > 1:
        raise ValueError(
            "Taxonomy embeddings must share one model and dimension"
        )
    version = stable_id(
        "taxonomy",
        cutoff.isoformat(),
        json_value(config),
        json_value(
            [
                {
                    "id": cid,
                    "label": known[cid].label,
                    "kind": known[cid].kind,
                    "vector": list(known[cid].vector),
                    "model": known[cid].embedding_model,
                    "first_seen": known[cid].first_seen.isoformat(),
                    "documents": sorted(
                        (doc, day.isoformat())
                        for doc, day in known[cid].document_evidence
                    ),
                    "dates": sorted(
                        day.isoformat() for day in known[cid].document_dates
                    ),
                }
                for cid in sorted(known)
            ]
        ),
        json_value(parents),
    )
    ids = sorted(known)
    vectors: Dict[str, np.ndarray] = {}
    if ids:
        matrix = _unit(
            np.array([known[cid].vector for cid in ids], dtype=float)
        )
        vectors = dict(zip(ids, matrix))
    # An explicit source hierarchy pulls the child toward its parent.
    weight = float(config["explicit_parent_weight"])
    clustered = dict(vectors)
    for child, parent in parents:
        if child in vectors and parent in vectors and child != parent:
            clustered[child] = vectors[child] + weight * vectors[parent]
            clustered[child] /= np.linalg.norm(clustered[child]) or 1.0

    nodes: Dict[str, TaxonomyNode] = {}
    placement: Dict[str, str] = {}
    general: set = set()
    minimum = int(config["min_cluster_size"])

    def split(
        members: List[str],
        path: Tuple[int, ...],
        parent_id: Optional[str],
    ) -> str:
        centroid = (
            _unit(
                np.array([clustered[cid] for cid in members]).sum(
                    axis=0, keepdims=True
                )
            )[0]
            if members
            else np.zeros(1)
        )
        node_id = stable_id("taxonomy-node", version, *path)
        node = TaxonomyNode(
            node_id=node_id,
            path=path,
            level=len(path),
            parent_id=parent_id,
            label=_label(
                members, known, clustered, centroid, config["label_terms"]
            )
            if members
            else "(empty)",
            centroid=centroid,
            subtree_ids=list(members),
        )
        nodes[node_id] = node
        k = min(int(config["branching"]), len(members) // minimum)
        if len(path) >= int(config["max_depth"]) or k < 2:
            node.concept_ids = list(members)
            return node_id
        matrix = np.array([clustered[cid] for cid in members])
        assignment, centroids = _spherical_kmeans(
            matrix,
            k,
            int(config["kmeans_iterations"]),
            int(config["seed"]) + len(path),
        )
        similarity = matrix @ centroids.T
        groups: Dict[int, List[str]] = {}
        for row, cid in enumerate(members):
            own = similarity[row, assignment[row]]
            others = np.delete(similarity[row], assignment[row])
            concentration = own - (others.max() if len(others) else -1.0)
            # TaxoGen: a term close to several children is general and
            # describes the parent itself.
            if concentration < float(config["general_term_margin"]):
                node.concept_ids.append(cid)
                general.add(cid)
            else:
                groups.setdefault(int(assignment[row]), []).append(cid)
        viable = [group for group in groups.values() if len(group) >= minimum]
        for group in groups.values():
            # Too few terms for a sub-topic: they describe the parent too.
            if len(group) < minimum:
                node.concept_ids.extend(group)
                general.update(group)
        if len(viable) < 2:
            node.concept_ids = list(members)
            general.difference_update(members)
            return node_id
        for index, group in enumerate(
            sorted(viable, key=lambda item: (-len(item), sorted(item)))
        ):
            node.children.append(split(sorted(group), (*path, index), node_id))
        return node_id

    split(ids, (), None)
    for node in nodes.values():
        for cid in node.concept_ids:
            placement[cid] = node.node_id

    year_ago = cutoff - timedelta(days=365)
    two_years_ago = cutoff - timedelta(days=730)
    previous_snapshot = cutoff - timedelta(days=int(config["known_age_days"]))
    for node in nodes.values():
        dated_documents = {}
        for cid in node.subtree_ids:
            concept = known[cid]
            evidence = concept.document_evidence or [
                (f"legacy:{cid}:{index}", value)
                for index, value in enumerate(concept.document_dates)
            ]
            for doc_id, value in evidence:
                dated_documents[doc_id] = min(
                    value, dated_documents.get(doc_id, value)
                )
        dates = list(dated_documents.values())
        node.documents_last_year = sum(value > year_ago for value in dates)
        node.documents_previous_year = sum(
            two_years_ago < value <= year_ago for value in dates
        )
        node.new_share = (
            sum(
                known[cid].first_seen > previous_snapshot
                for cid in node.subtree_ids
            )
            / len(node.subtree_ids)
            if node.subtree_ids
            else 0.0
        )
    return Taxonomy(
        version=version,
        snapshot=cutoff,
        nodes=nodes,
        placement=placement,
        general_terms=general,
        concepts=known,
    )


def taxonomy_features(
    taxonomy: Taxonomy, config: Optional[Dict[str, object]] = None
) -> Dict[str, Dict[str, object]]:
    """Novelty features of each placed concept at the taxonomy snapshot.

    - semantic_novelty: 1 - cosine to the nearest concept already known a
      year (``known_age_days``) before the snapshot;
    - new_branch_in_known_area: the concept's branch is mostly new while
      its parent branch is mostly established.
    """
    config = config or load_catalog("taxonomy")
    threshold = float(config["new_branch_share"])
    previous = taxonomy.snapshot - timedelta(
        days=int(config["known_age_days"])
    )
    ids = sorted(taxonomy.concepts)
    vectors = (
        _unit(np.array([taxonomy.concepts[cid].vector for cid in ids], float))
        if ids
        else np.zeros((0, 1))
    )
    established = np.array(
        [taxonomy.concepts[cid].first_seen <= previous for cid in ids]
    )
    features: Dict[str, Dict[str, object]] = {}
    for row, cid in enumerate(ids):
        node = taxonomy.nodes[taxonomy.placement[cid]]
        parent = taxonomy.nodes.get(node.parent_id) if node.parent_id else None
        mask = established.copy()
        mask[row] = False
        novelty = (
            1.0 - float(np.max(vectors[mask] @ vectors[row]))
            if mask.any()
            else 1.0
        )
        features[cid] = {
            "taxonomy_level": node.level,
            "taxonomy_node_size": len(node.subtree_ids),
            "taxonomy_sibling_count": len(parent.children) - 1
            if parent
            else 0,
            "taxonomy_general_term": cid in taxonomy.general_terms,
            "semantic_novelty": round(novelty, 6),
            "branch_growth": node.documents_last_year
            / max(1, node.documents_previous_year),
            "branch_new_share": round(node.new_share, 6),
            "new_branch_in_known_area": bool(
                parent is not None
                and node.new_share >= threshold
                and parent.new_share < threshold
            ),
        }
    return features
