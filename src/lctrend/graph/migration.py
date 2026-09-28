"""Migrate stored concepts to the lexical identity key v2.

Concepts created before key v2 carry mention-derived ids and no identity
key, and old lemmatization split one name into several concepts. The
migration recomputes every concept's key, seeds its form counts from its
mentions, and merges the concepts that key v2 identifies as one:

``lctrend migrate-concept-keys`` prints the plan; ``--apply`` writes the
keys and runs the merges (``graph.merge``). A second run finds nothing to
merge. Run it while no ingestion job is writing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from ..core.models import Concept, stable_id
from ..extraction.lexical import KEY_VERSION, kind_family
from ..extraction.resolver import _preferred, concept_identity

_KIND_RANK = {"Material": 1, "Method": 2, "Technology": 3}


@dataclass
class IdentityUpdate:
    concept_id: str
    kind: str
    identity_key: str
    label_counts: Dict[str, int]
    preferred_label: str


@dataclass
class ConceptMerge:
    source: str
    target: str
    identity_key: str
    source_label: str
    target_label: str


@dataclass
class MigrationPlan:
    updates: List[IdentityUpdate] = field(default_factory=list)
    merges: List[ConceptMerge] = field(default_factory=list)

    def summary(self, apply: bool) -> Dict[str, Any]:
        return {
            "key_version": KEY_VERSION,
            "apply": apply,
            "concepts": len(self.updates),
            "merges": len(self.merges),
            "merge_plan": [item.__dict__ for item in self.merges],
        }


def plan_key_migration(
    concepts: Sequence[Concept],
    mention_counts: Mapping[str, int] | None = None,
    form_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> MigrationPlan:
    """Group concepts by (kind family, key v2) and pick one target each.

    The target is, in order: a reviewed concept, the concept whose id
    key v2 would create, the most mentioned, the highest kind of the
    family, the smallest id — never the order concepts were read in.
    """
    mention_counts = mention_counts or {}
    form_counts = form_counts or {}
    plan = MigrationPlan()
    groups: Dict[tuple, List[Concept]] = {}
    for concept in concepts:
        key, kind = concept_identity(concept.preferred_label, concept.kind)
        family = kind_family(kind)
        groups.setdefault((family, key), []).append(concept)
        counts = dict(form_counts.get(concept.concept_id) or {})
        if not counts:
            counts = dict(concept.label_counts) or {concept.preferred_label: 1}
        plan.updates.append(
            IdentityUpdate(
                concept_id=concept.concept_id,
                kind=concept.kind.value,
                identity_key=key,
                label_counts=counts,
                preferred_label=concept.preferred_label
                if concept.status == "accepted"
                else _preferred(counts),
            )
        )
    for (family, key), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        canonical = stable_id("concept", family, key)
        target = min(
            members,
            key=lambda item: (
                item.status != "accepted",
                item.concept_id != canonical,
                -mention_counts.get(item.concept_id, 0),
                -_KIND_RANK.get(item.kind.value, 0),
                item.concept_id,
            ),
        )
        for member in sorted(members, key=lambda item: item.concept_id):
            if member is target:
                continue
            plan.merges.append(
                ConceptMerge(
                    source=member.concept_id,
                    target=target.concept_id,
                    identity_key=key,
                    source_label=member.preferred_label,
                    target_label=target.preferred_label,
                )
            )
    return plan


async def apply_key_migration(
    store: Any, apply: bool = False
) -> MigrationPlan:
    """Plan the migration of the store; with ``apply`` also execute it."""
    from ..core.aio import resolve

    concepts = await resolve(store.read_concepts())
    mention_counts, form_counts = await resolve(store.read_concept_forms())
    plan = plan_key_migration(concepts, mention_counts, form_counts)
    if not apply:
        return plan
    # Keys and counts first: a merge combines the targets' updated counts.
    await resolve(store.write_concept_identities(plan.updates))
    for item in plan.merges:
        await resolve(
            store.merge_concepts(
                item.source, item.target, reason=f"{KEY_VERSION} migration"
            )
        )
    return plan
