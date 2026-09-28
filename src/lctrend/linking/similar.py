"""SIMILAR_TO: a mutual nearest-neighbour layer over concept vectors.

Two concepts are similar when each is among the other's ``k`` nearest by
cosine of their stored vectors (resolver.concept_text, "label: definition"),
the cosine reaches ``min_cosine`` and both belong to one kind family
(Technology, Method and Material are one). Mutuality keeps umbrella names
from becoming the neighbour of everything.

The edges are computed, not reported: they carry ``method``, ``model``,
``cosine`` and ``computed_at``, and no evidence. ``observed_at`` is the day
the later of the two concepts first appeared, the earliest snapshot the
edge may belong to. Structural graph metrics (graph.novelty) read an
explicit list of evidence-backed relation types, so SIMILAR_TO never
enters PageRank, betweenness or neighbour entropy; it serves navigation,
the interface and, as its own edge type, graph learning.

A rebuild replaces every edge of the same method and model.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.config import cypher_identifier, load_catalog
from ..extraction.lexical import kind_family

logger = logging.getLogger(__name__)

METHOD = "mutual_knn"
RELATION = "SIMILAR_TO"
_BLOCK = 1024
_WRITE_BATCH = 1000

Edge = Tuple[str, str, float]


def settings() -> Dict[str, Any]:
    return dict(load_catalog("pipeline").get("linking", {}).get("similar", {}))


def mutual_neighbors(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    families: Sequence[str],
    k: int,
    min_cosine: float,
) -> List[Edge]:
    """(left, right, cosine) of mutual k-nearest pairs, left < right.

    Rows are compared in blocks, so the full n x n matrix never exists.
    """
    if len(ids) < 2 or k < 1:
        return []
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    family_codes = {name: code for code, name in enumerate(set(families))}
    codes = np.asarray([family_codes[name] for name in families])
    neighbors: List[Dict[int, float]] = []
    for start in range(0, len(ids), _BLOCK):
        block = matrix[start : start + _BLOCK] @ matrix.T
        rows = np.arange(block.shape[0])
        block[rows, rows + start] = -np.inf
        block[codes[start : start + _BLOCK, None] != codes[None, :]] = -np.inf
        size = min(k, len(ids) - 1)
        top = np.argpartition(-block, size - 1, axis=1)[:, :size]
        for row, columns in enumerate(top):
            neighbors.append(
                {
                    int(column): float(block[row, column])
                    for column in columns
                    if block[row, column] >= min_cosine
                }
            )
    edges = []
    for left, found in enumerate(neighbors):
        for right, cosine in found.items():
            if left < right and left in neighbors[right]:
                pair = sorted((ids[left], ids[right]))
                edges.append((pair[0], pair[1], round(cosine, 6)))
    return sorted(edges)


async def read_vectors(
    store: Any, model: str, kinds: Sequence[str]
) -> List[Dict[str, Any]]:
    """Active concepts of ``kinds`` with a vector of ``model``."""
    from ..graph.store import _records

    query = "\nUNION\n".join(
        f"MATCH (c:{cypher_identifier(kind)}) "
        "WHERE c.concept_id IS NOT NULL AND c.kind = $kinds["
        + str(index)
        + "] "
        "AND c.embedding IS NOT NULL AND c.embedding_model = $model "
        "AND NOT coalesce(c.status, '') IN ['merged', 'rejected'] "
        "RETURN c.concept_id AS concept_id, c.kind AS kind, "
        "c.embedding AS vector, c.first_seen_at AS first_seen_at"
        for index, kind in enumerate(kinds)
    )
    async with store._driver.session(database=store._database) as session:
        records = await _records(
            session, query, model=model, kinds=list(kinds)
        )
    return [
        {
            "concept_id": record["concept_id"],
            "kind": record["kind"],
            "vector": list(record["vector"]),
            "first_seen_at": record["first_seen_at"],
        }
        for record in records
    ]


def _later(left: Optional[str], right: Optional[str]) -> Optional[str]:
    """The edge exists once both concepts do; unknown stays unknown."""
    if left is None or right is None:
        return None
    return max(str(left), str(right))


async def write_edges(
    store: Any,
    edges: List[Edge],
    rows: Dict[str, Dict[str, Any]],
    model: str,
    computed_at: str,
) -> None:
    """Replace the SIMILAR_TO edges of this method and model."""
    from ..graph.store import _run

    # Grouped by the kinds of both ends, so each MATCH uses a label and its
    # concept_id index instead of scanning every node.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for left, right, cosine in edges:
        groups.setdefault(
            (rows[left]["kind"], rows[right]["kind"]), []
        ).append(
            {
                "left": left,
                "right": right,
                "cosine": cosine,
                "observed_at": _later(
                    rows[left]["first_seen_at"], rows[right]["first_seen_at"]
                ),
            }
        )
    async with store._driver.session(database=store._database) as session:

        async def replace(tx):
            await _run(
                tx,
                f"MATCH ()-[r:{RELATION} {{method: $method, model: $model}}]"
                "->() DELETE r",
                method=METHOD,
                model=model,
            )
            for (left_kind, right_kind), payload in sorted(groups.items()):
                for start in range(0, len(payload), _WRITE_BATCH):
                    await _run(
                        tx,
                        f"""
                        UNWIND $rows AS row
                        MATCH (a:{cypher_identifier(left_kind)}
                               {{concept_id: row.left}})
                        MATCH (b:{cypher_identifier(right_kind)}
                               {{concept_id: row.right}})
                        MERGE (a)-[r:{RELATION} {{method: $method,
                                                  model: $model}}]->(b)
                        SET r.cosine = row.cosine,
                            r.observed_at = row.observed_at,
                            r.computed_at = $computed_at
                        """,
                        rows=payload[start : start + _WRITE_BATCH],
                        method=METHOD,
                        model=model,
                        computed_at=computed_at,
                    )

        await session.execute_write(replace)


async def rebuild(
    store: Any,
    model: str,
    k: Optional[int] = None,
    min_cosine: Optional[float] = None,
    kinds: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Recompute and store the SIMILAR_TO layer of one embedding model."""
    options = settings()
    k = int(k if k is not None else options.get("k", 8))
    min_cosine = float(
        min_cosine
        if min_cosine is not None
        else options.get("min_cosine", 0.7)
    )
    kinds = list(
        kinds or options.get("kinds") or ["Technology", "Method", "Material"]
    )
    rows = await read_vectors(store, model, kinds)
    edges = mutual_neighbors(
        [row["concept_id"] for row in rows],
        [row["vector"] for row in rows],
        [kind_family(row["kind"]) for row in rows],
        k,
        min_cosine,
    )
    computed_at = datetime.now(timezone.utc).isoformat()
    if not dry_run:
        await write_edges(
            store,
            edges,
            {row["concept_id"]: row for row in rows},
            model,
            computed_at,
        )
    degrees: Dict[str, int] = {}
    for left, right, _ in edges:
        degrees[left] = degrees.get(left, 0) + 1
        degrees[right] = degrees.get(right, 0) + 1
    summary = {
        "model": model,
        "method": METHOD,
        "k": k,
        "min_cosine": min_cosine,
        "kinds": kinds,
        "concepts": len(rows),
        "edges": len(edges),
        "linked_concepts": len(degrees),
        "max_degree": max(degrees.values(), default=0),
        "computed_at": computed_at,
        "written": not dry_run,
    }
    logger.info("SIMILAR_TO rebuilt: %s", summary)
    return summary
