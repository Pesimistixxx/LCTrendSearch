"""Concept identity does not depend on the order of documents (C-4)."""

from itertools import permutations

from lctrend.core.models import ConceptKind, Mention
from lctrend.extraction.resolver import ConceptIndex, resolve_mentions

T = ConceptKind.TECHNOLOGY
METHOD = ConceptKind.METHOD
MATERIAL = ConceptKind.MATERIAL

DOCUMENTS = [
    ("d1", [("LLM", T), ("federated learning", METHOD)]),
    ("d2", [("большие языковые модели", T), ("graphene", MATERIAL)]),
    ("d3", [("больших языковых моделей", T), ("Graphene", T)]),
    ("d4", [("большие языковые модели", T), ("federated learning", T)]),
    ("d5", [("квантовый отжиг", T), ("квантового отжига", T)]),
]


def mentions(document_id, names):
    return [
        Mention(
            mention_id=f"{document_id}:{index}",
            chunk_id=f"{document_id}:c1",
            surface_text=text,
            canonical_text=text,
            start=0,
            end=len(text),
            type_candidates=[kind],
        )
        for index, (text, kind) in enumerate(names)
    ]


def resolve_in_order(documents):
    index = ConceptIndex()
    bindings = {}
    for document_id, names in documents:
        _, decisions = resolve_mentions(mentions(document_id, names), index)
        for decision in decisions:
            bindings[decision.mention_id] = decision.concept_id
    concepts = {
        (concept.identity_key, concept.kind, concept.concept_id)
        for concept in index
    }
    labels = {concept.concept_id: concept.preferred_label for concept in index}
    return concepts, labels, bindings


def test_document_order_does_not_change_key_kind_or_id():
    expected = resolve_in_order(DOCUMENTS)
    for order in permutations(DOCUMENTS):
        assert resolve_in_order(order) == expected
    concepts, labels, _ = expected
    assert len(concepts) == 6
    kinds = {kind for _, kind, _ in concepts}
    # Lexical proposals preserve their exact types. A lone type annotation
    # cannot promote a method or material; publication needs a profile.
    assert kinds == {T, METHOD, MATERIAL}
    assert "большие языковые модели" in labels.values()


def test_concept_id_preserves_kind_instead_of_promoting():
    # Conflicting type proposals stay separate until semantic review.
    first, _, _ = resolve_in_order([("a", [("квантовый отжиг", T)])])
    second, _, _ = resolve_in_order([("b", [("квантового отжига", METHOD)])])
    assert {(key, cid) for key, _, cid in first} != {
        (key, cid) for key, _, cid in second
    }


def test_preferred_label_is_the_most_frequent_form():
    documents = [
        ("a", [("Quantum annealer", T)]),
        ("b", [("quantum annealers", T)]),
        ("c", [("quantum annealers", T)]),
    ]
    for order in permutations(documents):
        _, labels, _ = resolve_in_order(order)
        assert list(labels.values()) == ["quantum annealers"]
