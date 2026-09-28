"""Merging duplicate concepts of one identity family.

The source concept keeps its node for the audit, marked ``merged`` and
linked ``MERGED_INTO`` the target. Its mentions, assertion roles,
projections and evidence links move to the target, and its names become
accepted names of the target, so later documents resolve them there.
Merged concepts are left out of the registry and the feature reads.

Merges are meant to run while no ingestion job is writing: a running job
keeps its in-memory registry until it finishes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.config import cypher_identifier, load_catalog
from ..core.models import (
    Concept,
    ConceptKind,
    ConceptName,
    json_value,
    stable_id,
)
from ..extraction.lexical import kind_family, settled_kind
from ..extraction.resolver import _preferred, normalize_name

logger = logging.getLogger(__name__)

MERGE_METHOD = "concept_merge"
# Keyed relationships are re-created with MERGE on their key, so a link the
# target already has is updated instead of duplicated.
_EVIDENCE = (
    ("HAS_MATURITY_EVIDENCE", ("assertion_id", "start")),
    ("HAS_ECONOMIC_EVIDENCE", ("evidence_id",)),
)


def merged_concept(target: Concept, source: Concept) -> Concept:
    """The target after absorbing the source; a merge is a review."""
    names: List[ConceptName] = []
    seen = set()
    for name in target.names:
        names.append(name)
        seen.add(normalize_name(name.text))
    for text in [source.preferred_label, *(n.text for n in source.names)]:
        normalized = normalize_name(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        names.append(
            ConceptName(
                name_id=stable_id("name", target.concept_id, normalized),
                text=text,
                normalized_text=normalized,
                name_kind="merged",
                status="accepted",
            )
        )
    counts = dict(target.label_counts or {target.preferred_label: 1})
    for form, count in (
        source.label_counts or {source.preferred_label: 1}
    ).items():
        counts[form] = counts.get(form, 0) + count
    kinds: Dict[str, int] = {}
    for concept in (target, source):
        for kind, count in (
            concept.kind_counts or {concept.kind.value: 1}
        ).items():
            kinds[kind] = kinds.get(kind, 0) + count
    if kind_family(target.kind) == "technology" and source.kind != target.kind:
        raise ValueError(
            "Different entity kinds require semantic reprocessing"
        )
    if source.identity_scope != target.identity_scope:
        raise ValueError("Different entity meanings cannot be merged")
    kind = (
        target.kind.value
        if kind_family(target.kind) == "technology"
        else settled_kind(kinds, target.kind)
    )
    return target.model_copy(
        update={
            "kind": ConceptKind(kind),
            "names": names,
            "label_counts": counts,
            "kind_counts": kinds,
            "preferred_label": target.preferred_label
            if target.status == "accepted"
            else _preferred(counts),
            "definition": target.definition or source.definition,
            "identity_key": target.identity_key or source.identity_key,
        },
        deep=True,
    )


def _relationship_moves(source: str, target: str) -> List[str]:
    """Statements that move every catalogued link from source to target."""
    graph = load_catalog("graph")
    s = f"(s:{source} {{concept_id: $source}})"
    t = f"(t:{target} {{concept_id: $target}})"
    statements = [
        f"""
        MATCH {s}<-[r:MENTIONS]-(chunk:Chunk)
        WHERE r.mention_id IS NOT NULL
        MATCH {t}
        MERGE (chunk)-[moved:MENTIONS {{mention_id: r.mention_id}}]->(t)
        SET moved += properties(r), moved.merged_from = $source
        DELETE r
        """,
        # A collision between exactly these two concepts is now resolved.
        f"""
        MATCH {t}<-[m:MENTIONS]-(chunk:Chunk)
        WHERE m.resolution_status = 'ambiguous'
        OPTIONAL MATCH (chunk)-[other:MENTIONS]->(o)
        WHERE other.mention_id = m.mention_id AND o <> t
        WITH m, count(other) AS others
        WHERE others = 0
        SET m.resolution_status = 'accepted', m.method = '{MERGE_METHOD}'
        """,
    ]
    for relation in dict.fromkeys(graph["assertion_roles"].values()):
        relation = cypher_identifier(relation)
        statements.append(
            f"""
            MATCH (a:Assertion)-[r:{relation}]->{s}
            MATCH {t}
            MERGE (a)-[:{relation}]->(t)
            DELETE r
            """
        )
    projections = [
        cypher_identifier(item["relationship"])
        for item in graph["projections"].values()
    ]
    for relation in dict.fromkeys(projections):
        key = "{document_version_id: r.document_version_id}"
        statements += [
            f"""
            MATCH {s}-[r:{relation}]->(x)
            MATCH {t}
            FOREACH (_ IN CASE WHEN x <> t THEN [1] ELSE [] END |
                MERGE (t)-[moved:{relation} {key}]->(x)
                SET moved += properties(r))
            DELETE r
            """,
            f"""
            MATCH (x)-[r:{relation}]->{s}
            MATCH {t}
            FOREACH (_ IN CASE WHEN x <> t THEN [1] ELSE [] END |
                MERGE (x)-[moved:{relation} {key}]->(t)
                SET moved += properties(r))
            DELETE r
            """,
        ]
    for relation, keys in _EVIDENCE:
        key = ", ".join(f"{name}: r.{name}" for name in keys)
        statements.append(
            f"""
            MATCH {s}-[r:{relation}]->(chunk:Chunk)
            MATCH {t}
            MERGE (t)-[moved:{relation} {{{key}}}]->(chunk)
            SET moved += properties(r)
            DELETE r
            """
        )
    statements += [
        f"""
        MATCH {s}-[r:SAME_AS]->(x)
        MATCH {t}
        FOREACH (_ IN CASE WHEN r.document_version_id IS NULL
                 THEN [1] ELSE [] END |
            MERGE (t)-[moved:SAME_AS]->(x) SET moved += properties(r))
        FOREACH (_ IN CASE WHEN r.document_version_id IS NOT NULL
                 THEN [1] ELSE [] END |
            MERGE (t)-[moved:SAME_AS
                  {{document_version_id: r.document_version_id}}]->(x)
            SET moved += properties(r))
        DELETE r
        """,
        f"""
        MATCH {s}-[r:POSSIBLY_SAME_AS]-(x)
        DELETE r
        """,
        # The placement of a merged concept is stale until the taxonomy is
        # rebuilt.
        f"""
        MATCH {s}-[r:IN_TAXONOMY]->()
        DELETE r
        """,
    ]
    return statements


async def _merge(
    tx: Any,
    source: Concept,
    target: Concept,
    merged: Concept,
    reason: Optional[str],
    merged_at: str,
) -> None:
    from .store import _run

    if merged.kind != target.kind:
        await _run(
            tx,
            f"""
            MATCH (c:{target.kind.value} {{concept_id: $target}})
            REMOVE c:{target.kind.value}
            SET c:{merged.kind.value}, c.kind = $kind
            """,
            target=target.concept_id,
            kind=merged.kind.value,
        )
    label = cypher_identifier(merged.kind.value)
    for statement in _relationship_moves(
        cypher_identifier(source.kind.value), label
    ):
        await _run(
            tx,
            statement,
            source=source.concept_id,
            target=target.concept_id,
        )
    await _run(
        tx,
        f"""
        MATCH (s:{source.kind.value} {{concept_id: $source}})
        MATCH (t:{label} {{concept_id: $target}})
        SET t.names_json = $names_json, t.aliases = $aliases,
            t.normalized_aliases = $normalized_aliases,
            t.preferred_label = $preferred_label, t.name = $preferred_label,
            t.label_counts_json = $label_counts_json,
            t.kind_counts_json = $kind_counts_json,
            t.definition = coalesce(t.definition, $definition),
            t.identity_key = $identity_key,
            t.first_seen_at = CASE
                WHEN s.first_seen_at IS NULL THEN t.first_seen_at
                WHEN t.first_seen_at IS NULL
                    OR s.first_seen_at < t.first_seen_at
                THEN s.first_seen_at
                ELSE t.first_seen_at END
        """,
        source=source.concept_id,
        target=target.concept_id,
        names_json=json_value([name.model_dump() for name in merged.names]),
        aliases=list(
            dict.fromkeys(
                [
                    merged.preferred_label,
                    *(
                        name.text
                        for name in merged.names
                        if name.status == "accepted"
                    ),
                ]
            )
        ),
        normalized_aliases=list(
            dict.fromkeys(
                name.normalized_text
                for name in merged.names
                if name.status == "accepted"
            )
        ),
        preferred_label=merged.preferred_label,
        label_counts_json=json_value(merged.label_counts),
        kind_counts_json=json_value(merged.kind_counts),
        definition=merged.definition,
        identity_key=merged.identity_key,
    )
    await _run(
        tx,
        f"""
        MATCH (s:{source.kind.value} {{concept_id: $source}})
        MATCH (t:{label} {{concept_id: $target}})
        SET s.status = 'merged', s.merged_into = $target,
            s.merged_at = $merged_at
        MERGE (s)-[m:MERGED_INTO]->(t)
        SET m.reason = $reason, m.merged_at = $merged_at,
            m.method = '{MERGE_METHOD}'
        """,
        source=source.concept_id,
        target=target.concept_id,
        reason=reason,
        merged_at=merged_at,
    )


async def read_concepts_by_id(store: Any, ids: List[str]) -> Dict[str, Any]:
    from .store import CONCEPT_LABELS, _concept_from_properties, _records

    query = "\nUNION\n".join(
        f"UNWIND $ids AS id MATCH (c:{cypher_identifier(label)} "
        "{concept_id: id}) WHERE c.kind IS NOT NULL "
        "RETURN properties(c) AS properties"
        for label in CONCEPT_LABELS
    )
    async with store._driver.session(database=store._database) as session:
        records = await _records(session, query, ids=ids)
    found: Dict[str, Any] = {}
    for record in records:
        properties = record["properties"]
        found[properties["concept_id"]] = (
            _concept_from_properties(properties),
            properties.get("status"),
        )
    return found


async def merge_concepts(
    store: Any,
    source_id: str,
    target_id: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge ``source_id`` into ``target_id`` in one write transaction."""
    from ..core.aio import resolve

    if source_id == target_id:
        raise ValueError("a concept cannot be merged into itself")
    found = await read_concepts_by_id(store, [source_id, target_id])
    for concept_id in (source_id, target_id):
        if concept_id not in found:
            raise ValueError(f"concept {concept_id} not found")
        if found[concept_id][1] == "merged":
            raise ValueError(f"concept {concept_id} is already merged")
    source, target = found[source_id][0], found[target_id][0]
    if kind_family(source.kind) != kind_family(target.kind):
        raise ValueError(
            f"{source.kind.value} and {target.kind.value} are not one "
            "identity family"
        )
    merged = merged_concept(target, source)
    merged_at = datetime.now(timezone.utc).isoformat()
    async with store._driver.session(database=store._database) as session:
        await resolve(
            session.execute_write(
                _merge, source, target, merged, reason, merged_at
            )
        )
    logger.info(
        "Merged %s (%s) into %s (%s)",
        source_id,
        source.preferred_label,
        target_id,
        merged.preferred_label,
    )
    return {
        "source": source_id,
        "target": target_id,
        "target_kind": merged.kind.value,
        "preferred_label": merged.preferred_label,
        "merged_at": merged_at,
    }
