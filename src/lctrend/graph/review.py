"""Batch review of merge candidates (POSSIBLY_SAME_AS).

A candidate pair comes from the source itself (an alias it declared that
already names another concept) or from semantic similarity. Each pair is
shown with what the graph knows of both concepts: definition, names,
mentions, domains, parent technologies. ``lctrend review-duplicates``
prints the plan; ``--apply`` merges

- aliases a source declared: the text equated the names;
- semantic pairs at or above ``--merge-above``, only when asked;

and never a pair whose concepts belong to disjoint domains: "Transformer"
of machine learning and of power engineering share a name, not a meaning.
Every other pair stays pending for a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.models import DECLARED_ALIAS_METHOD
from ..extraction.lexical import kind_family


@dataclass
class CandidatePair:
    source: Dict[str, Any]
    target: Dict[str, Any]
    method: str
    score: Optional[float] = None
    cosine: Optional[float] = None
    alias: Optional[str] = None

    @property
    def domains_disjoint(self) -> bool:
        left = set(self.source.get("domains") or [])
        right = set(self.target.get("domains") or [])
        return bool(left and right and not left & right)


@dataclass
class ReviewDecision:
    pair: CandidatePair
    action: str  # merge | review
    reason: str
    merge_source: Optional[str] = None
    merge_target: Optional[str] = None


@dataclass
class ReviewPlan:
    decisions: List[ReviewDecision] = field(default_factory=list)

    def summary(self, apply: bool) -> Dict[str, Any]:
        def concept(item: Dict[str, Any]) -> Dict[str, Any]:
            return {
                key: item.get(key)
                for key in (
                    "concept_id",
                    "label",
                    "kind",
                    "definition",
                    "mentions",
                    "domains",
                    "parents",
                )
            }

        return {
            "apply": apply,
            "candidates": len(self.decisions),
            "merges": sum(d.action == "merge" for d in self.decisions),
            "pending": sum(d.action == "review" for d in self.decisions),
            "pairs": [
                {
                    "action": decision.action,
                    "reason": decision.reason,
                    "method": decision.pair.method,
                    "score": decision.pair.score,
                    "alias": decision.pair.alias,
                    "source": concept(decision.pair.source),
                    "target": concept(decision.pair.target),
                    "merge": [decision.merge_source, decision.merge_target]
                    if decision.action == "merge"
                    else None,
                }
                for decision in self.decisions
            ],
        }


def _keeper(pair: CandidatePair) -> tuple:
    """The concept that remains: reviewed, then most mentioned, then the
    smallest id."""
    left, right = pair.source, pair.target

    def rank(item: Dict[str, Any]) -> tuple:
        return (
            item.get("status") != "accepted",
            -int(item.get("mentions") or 0),
            str(item["concept_id"]),
        )

    keep, drop = sorted((left, right), key=rank)
    return drop["concept_id"], keep["concept_id"]


def plan_review(
    pairs: Sequence[CandidatePair], merge_above: Optional[float] = None
) -> ReviewPlan:
    plan = ReviewPlan()
    for pair in sorted(
        pairs,
        key=lambda item: (
            item.method != DECLARED_ALIAS_METHOD,
            -(item.score or 0.0),
            item.source["concept_id"],
            item.target["concept_id"],
        ),
    ):
        if kind_family(pair.source["kind"]) != kind_family(
            pair.target["kind"]
        ):
            action, reason = "review", "different kind families"
        elif pair.domains_disjoint:
            action, reason = "review", "disjoint domains: possible homonym"
        elif pair.method == DECLARED_ALIAS_METHOD:
            action, reason = "merge", f"source declared alias {pair.alias!r}"
        elif (
            merge_above is not None
            and pair.score is not None
            and pair.score >= merge_above
        ):
            action, reason = (
                "merge", f"score {pair.score:.3f} >= {merge_above}"
            )
        else:
            action, reason = "review", "semantic similarity only"
        decision = ReviewDecision(pair, action, reason)
        if action == "merge":
            decision.merge_source, decision.merge_target = _keeper(pair)
        plan.decisions.append(decision)
    return plan


async def review_duplicates(
    store: Any,
    apply: bool = False,
    merge_above: Optional[float] = None,
    limit: Optional[int] = None,
) -> ReviewPlan:
    """Plan the review of pending candidates; with ``apply`` merge the
    pairs the plan merges. A chain (A~B, B~C) follows earlier merges."""
    from ..core.aio import resolve

    rows = await resolve(store.read_merge_candidates(limit))
    pairs = [
        CandidatePair(
            source=row["source"],
            target=row["target"],
            method=row.get("method") or "",
            score=row.get("score"),
            cosine=row.get("cosine"),
            alias=row.get("alias"),
        )
        for row in rows
    ]
    plan = plan_review(pairs, merge_above)
    if not apply:
        return plan
    merged: Dict[str, str] = {}

    def current(concept_id: str) -> str:
        while concept_id in merged:
            concept_id = merged[concept_id]
        return concept_id

    for decision in plan.decisions:
        if decision.action != "merge":
            continue
        source = current(decision.merge_source)
        target = current(decision.merge_target)
        if source == target:
            continue
        await resolve(
            store.merge_concepts(
                source, target, reason=f"review: {decision.reason}"
            )
        )
        merged[source] = target
    return plan
