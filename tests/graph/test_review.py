"""Batch review of merge candidates with their graph context."""

import asyncio

from lctrend.graph.review import CandidatePair, plan_review, review_duplicates


def concept(concept_id, label, kind="Material", mentions=1, **extra):
    return {
        "concept_id": concept_id,
        "label": label,
        "kind": kind,
        "mentions": mentions,
        **extra,
    }


def test_a_declared_alias_merges_into_the_most_mentioned_concept():
    pair = CandidatePair(
        concept("c:compactin", "compactin", mentions=1),
        concept("c:ml", "ML-236B", mentions=5),
        "declared_alias",
        1.0,
        alias="ML-236B",
    )
    (decision,) = plan_review([pair]).decisions
    assert decision.action == "merge"
    assert (decision.merge_source, decision.merge_target) == (
        "c:compactin",
        "c:ml",
    )


def test_semantic_pairs_merge_only_above_an_explicit_threshold():
    pair = CandidatePair(
        concept("c:a", "LLM agents", "Technology"),
        concept("c:b", "agentic LLM", "Technology"),
        "new_provisional_semantic_candidate",
        0.91,
    )
    assert plan_review([pair]).decisions[0].action == "review"
    assert plan_review([pair], merge_above=0.9).decisions[0].action == "merge"


def test_disjoint_domains_or_families_are_never_merged():
    homonym = CandidatePair(
        concept("c:t1", "Transformer", "Technology", domains=["NLP"]),
        concept(
            "c:t2", "transformer", "Technology", domains=["Power grids"]
        ),
        "declared_alias",
        1.0,
    )
    families = CandidatePair(
        concept("c:m", "Merck", "Company"),
        concept("c:l", "lovastatin"),
        "declared_alias",
        1.0,
    )
    plan = plan_review([homonym, families], merge_above=0.0)
    assert [d.action for d in plan.decisions] == ["review", "review"]
    assert "homonym" in plan.decisions[1].reason + plan.decisions[0].reason


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.merges = []

    async def read_merge_candidates(self, limit=None):
        return self.rows

    async def merge_concepts(self, source, target, reason=None):
        self.merges.append((source, target))


def row(source, target):
    return {
        "source": concept(source, source, mentions=1),
        "target": concept(target, target, mentions=9 if target == "c" else 2),
        "method": "declared_alias",
        "score": 1.0,
        "alias": target,
    }


def test_apply_follows_a_chain_of_merges():
    store = Store([row("a", "b"), row("b", "c"), row("a", "c")])
    plan = asyncio.run(review_duplicates(store))
    assert store.merges == [] and plan.summary(False)["merges"] == 3
    asyncio.run(review_duplicates(store, apply=True))
    # a→b, then b→c; a→c is already done through b.
    assert store.merges == [("a", "b"), ("b", "c")]
