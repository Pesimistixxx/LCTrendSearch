"""Bring a stored graph to the current naming of countries and
organizations.

Graphs written before this naming keep country codes as names ("US"),
one company per OpenAlex country ("Intel (United States)", "Intel
(Germany)") and sponsors typed by their role ("Samsung" as a funder, not a
company). New documents are written the new way; this migration fixes the
nodes already stored:

``lctrend normalize-graph`` prints the plan; ``--apply`` writes it. A
second run finds nothing to change. Run it while no ingestion job is
writing, then ``lctrend migrate-concept-keys --apply`` to merge concepts
of one organization that the text typed differently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..core.config import cypher_identifier
from ..core.organizations import (
    country_names,
    display_rank,
    organization_identity,
    organization_type,
    source_organization_type,
)

_CODE = re.compile(r"[A-Z]{2}")
_LABELS = {"company": "Company", "university": "University"}


@dataclass
class OrganizationGroup:
    """Stored organizations that are one organization now."""

    organization_id: str
    name: str
    organization_type: str
    members: List[str]


@dataclass
class ConceptRelabel:
    concept_id: str
    source: str
    target: str
    label: str


@dataclass
class NormalizationPlan:
    countries: List[Dict[str, Any]] = field(default_factory=list)
    organizations: List[OrganizationGroup] = field(default_factory=list)
    concepts: List[ConceptRelabel] = field(default_factory=list)

    def summary(self, apply: bool) -> Dict[str, Any]:
        return {
            "apply": apply,
            "countries_renamed": len(self.countries),
            "organizations_updated": len(self.organizations),
            "organizations_merged_away": sum(
                len(group.members) - 1 for group in self.organizations
            ),
            "concepts_relabeled": len(self.concepts),
            "organization_plan": [
                {
                    "name": group.name,
                    "type": group.organization_type,
                    "merges": len(group.members),
                }
                for group in self.organizations
            ],
            "concept_plan": [item.__dict__ for item in self.concepts],
        }


def plan_countries(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Country nodes whose name is not the full name of their code."""
    plan = []
    for row in rows:
        code = row.get("code") or row.get("preferred_label") or ""
        if not _CODE.fullmatch(code):
            continue
        name, name_en = country_names(code)
        if row.get("name") != name or row.get("name_en") != name_en:
            plan.append(
                {
                    "element_id": row["element_id"],
                    "code": code,
                    "name": name,
                    "name_en": name_en,
                }
            )
    return plan


def plan_organizations(
    rows: Sequence[Dict[str, Any]],
) -> List[OrganizationGroup]:
    """Group metadata organizations by their current identity.

    A company's identity is its name without country and legal form, so
    OpenAlex's per-country copies become one node; the sponsor type is
    re-read from the name. The target keeps the id the adapters now
    write, so later documents reach the same node.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        kind = source_organization_type(
            row["name"], row.get("organization_type")
        )
        if kind == "company":
            new_id, name = organization_identity(row["name"], kind, "", "")
        else:
            new_id, name = row["organization_id"], row["name"]
        groups.setdefault(new_id, []).append(
            {**row, "new_type": kind, "new_name": name}
        )
    plan = []
    for new_id, members in sorted(groups.items()):
        best = min(
            members,
            key=lambda item: (
                display_rank(item["new_name"]),
                item["new_name"],
            ),
        )
        kind = best["new_type"]
        unchanged = (
            len(members) == 1
            and members[0]["organization_id"] == new_id
            and members[0]["name"] == best["new_name"]
            and members[0].get("organization_type") == kind
            and (
                kind not in _LABELS
                or _LABELS[kind] in (members[0].get("labels") or [])
            )
        )
        if unchanged:
            continue
        plan.append(
            OrganizationGroup(
                organization_id=new_id,
                name=best["new_name"],
                organization_type=kind,
                members=sorted(item["organization_id"] for item in members),
            )
        )
    return plan


def plan_concepts(rows: Sequence[Dict[str, Any]]) -> List[ConceptRelabel]:
    """Organization concepts whose name the catalog types differently."""
    plan = []
    for row in rows:
        target = _LABELS.get(organization_type(row["preferred_label"], ""))
        if target and target != row["kind"]:
            plan.append(
                ConceptRelabel(
                    concept_id=row["concept_id"],
                    source=row["kind"],
                    target=target,
                    label=row["preferred_label"],
                )
            )
    return plan


async def _read(store: Any, query: str) -> List[Dict[str, Any]]:
    from .store import _data, _records

    async with store._driver.session(database=store._database) as session:
        return [_data(record) for record in await _records(session, query)]


async def read_plan(store: Any) -> NormalizationPlan:
    countries = await _read(
        store,
        """
        MATCH (c:Country)
        RETURN elementId(c) AS element_id, c.code AS code, c.name AS name,
               c.name_en AS name_en, c.preferred_label AS preferred_label
        """,
    )
    organizations = await _read(
        store,
        """
        MATCH (o:Organization)
        WHERE o.organization_id IS NOT NULL AND o.concept_id IS NULL
        RETURN o.organization_id AS organization_id, o.name AS name,
               o.organization_type AS organization_type,
               labels(o) AS labels
        """,
    )
    concepts = await _read(
        store,
        """
        MATCH (c)
        WHERE (c:Organization OR c:Company OR c:University)
          AND c.concept_id IS NOT NULL AND c.kind IS NOT NULL
          AND coalesce(c.status, '') <> 'merged'
        RETURN c.concept_id AS concept_id, c.kind AS kind,
               c.preferred_label AS preferred_label
        """,
    )
    return NormalizationPlan(
        countries=plan_countries(countries),
        organizations=plan_organizations(
            [row for row in organizations if row.get("name")]
        ),
        concepts=plan_concepts(concepts),
    )


async def _relationship_types(tx: Any, organization_id: str) -> List[tuple]:
    from .store import _data, _records

    rows = await _records(
        tx,
        """
        MATCH (o:Organization {organization_id: $id})-[r]-(other)
        RETURN DISTINCT type(r) AS type, startNode(r) = o AS outgoing
        """,
        id=organization_id,
    )
    return [(row["type"], row["outgoing"]) for row in map(_data, rows)]


async def _merge_group(tx: Any, group: OrganizationGroup) -> None:
    """Fold the group into one node carrying the group's new id."""
    from .store import _run, _single

    existing = await _single(
        tx,
        "OPTIONAL MATCH (o:Organization {organization_id: $id}) "
        "RETURN o IS NOT NULL AS found",
        id=group.organization_id,
    )
    target = (
        group.organization_id
        if existing and existing["found"]
        else group.members[0]
    )
    for source in group.members:
        if source == target:
            continue
        for relation, outgoing in await _relationship_types(tx, source):
            relation = cypher_identifier(relation)
            pattern = (
                f"(s)-[r:{relation}]->(x)"
                if outgoing
                else f"(x)-[r:{relation}]->(s)"
            )
            moved = (
                f"(t)-[n:{relation}]->(x)"
                if outgoing
                else f"(x)-[n:{relation}]->(t)"
            )
            await _run(
                tx,
                f"""
                MATCH (s:Organization {{organization_id: $source}})
                MATCH (t:Organization {{organization_id: $target}})
                MATCH {pattern}
                WHERE x <> t
                MERGE {moved}
                SET n += properties(r)
                DELETE r
                """,
                source=source,
                target=target,
            )
        await _run(
            tx,
            """
            MATCH (s:Organization {organization_id: $source})
            MATCH (t:Organization {organization_id: $target})
            SET t.external_ids = coalesce(t.external_ids, [])
                + [item IN coalesce(s.external_ids, [])
                   WHERE NOT item IN coalesce(t.external_ids, [])],
                t.first_seen_at = CASE
                    WHEN s.first_seen_at IS NOT NULL
                        AND (t.first_seen_at IS NULL
                             OR s.first_seen_at < t.first_seen_at)
                    THEN s.first_seen_at ELSE t.first_seen_at END
            DETACH DELETE s
            """,
            source=source,
            target=target,
        )
    label = _LABELS.get(group.organization_type)
    await _run(
        tx,
        f"""
        MATCH (t:Organization {{organization_id: $target}})
        SET t.organization_id = $id, t.name = $name,
            t.name_rank = $name_rank,
            t.organization_type = $type
        {f"SET t:{label}" if label else ""}
        """,
        target=target,
        id=group.organization_id,
        name=group.name,
        name_rank=display_rank(group.name),
        type=group.organization_type,
    )
    # Versions keep their parties for point-in-time features.
    mapping = {member: group.organization_id for member in group.members}
    for property_name in ("organization_ids", "company_ids", "university_ids"):
        await _run(
            tx,
            f"""
            MATCH (v:DocumentVersion)
            WHERE any(item IN coalesce(v.{property_name}, [])
                      WHERE item IN $members)
            SET v.{property_name} = reduce(
                acc = [], item IN v.{property_name} |
                CASE WHEN coalesce($mapping[item], item) IN acc THEN acc
                     ELSE acc + coalesce($mapping[item], item) END)
            """,
            members=group.members,
            mapping=mapping,
        )
    if group.organization_type in ("company", "university"):
        property_name = f"{group.organization_type}_ids"
        await _run(
            tx,
            f"""
            MATCH (v:DocumentVersion)
            WHERE $id IN coalesce(v.organization_ids, [])
              AND NOT $id IN coalesce(v.{property_name}, [])
            SET v.{property_name} = coalesce(v.{property_name}, []) + $id
            """,
            id=group.organization_id,
        )


async def _write(tx: Any, plan: NormalizationPlan) -> None:
    from .store import _run

    if plan.countries:
        await _run(
            tx,
            """
            UNWIND $rows AS row
            MATCH (c:Country) WHERE elementId(c) = row.element_id
            SET c.name = row.name, c.name_en = row.name_en,
                c.code = row.code
            """,
            rows=plan.countries,
        )
    for group in plan.organizations:
        await _merge_group(tx, group)
    for item in plan.concepts:
        source, target = map(cypher_identifier, (item.source, item.target))
        await _run(
            tx,
            f"""
            MATCH (c:{source} {{concept_id: $id}})
            REMOVE c:{source}
            SET c:{target}, c.kind = $kind
            """,
            id=item.concept_id,
            kind=item.target,
        )


async def _link_companies(tx: Any) -> int:
    """SAME_AS from each company concept to the metadata company."""
    from .store import _data, _records, _run

    rows = [
        _data(record)
        for record in await _records(
            tx,
            """
            MATCH (c:Company)
            WHERE c.concept_id IS NOT NULL
              AND coalesce(c.status, '') <> 'merged'
            RETURN c.concept_id AS concept_id, c.preferred_label AS label
            """,
        )
    ]
    links = [
        {
            "concept_id": row["concept_id"],
            "organization_id": organization_identity(
                row["label"], "company", "", ""
            )[0],
        }
        for row in rows
        if row.get("label")
    ]
    if links:
        await _run(
            tx,
            """
            UNWIND $rows AS row
            MATCH (c:Company {concept_id: row.concept_id})
            MATCH (x:Organization {organization_id: row.organization_id})
            MERGE (c)-[:SAME_AS]->(x)
            """,
            rows=links,
        )
    return len(links)


async def normalize_graph(
    store: Any, apply: bool = False
) -> NormalizationPlan:
    """Plan the normalization of the store; with ``apply`` also write it."""
    from ..core.aio import resolve

    plan = await read_plan(store)
    if not apply:
        return plan

    async def write(tx):
        await _write(tx, plan)
        await _link_companies(tx)

    async with store._driver.session(database=store._database) as session:
        await resolve(session.execute_write(write))
    return plan
