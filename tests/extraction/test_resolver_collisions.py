"""Identity-key collisions are narrowed or kept, never dropped (C-3)."""

from lctrend.core.models import Concept, ConceptKind, Mention
from lctrend.extraction.resolver import resolve_mentions

T = ConceptKind.TECHNOLOGY


def duplicate(concept_id, label, kind=T):
    # Two graph concepts with one key: a CLI run and a web job created
    # them concurrently, or they predate key v2.
    return Concept(
        concept_id=concept_id,
        kind=kind,
        preferred_label=label,
        status="provisional",
    )


def mention(text, kind=T):
    return Mention(
        mention_id="m1",
        chunk_id="c1",
        surface_text=text,
        canonical_text=text,
        start=0,
        end=len(text),
        type_candidates=[kind],
    )


REGISTRY = [
    duplicate("concept:a", "квантовый отжиг"),
    duplicate("concept:b", "Квантового отжига"),
]


def test_an_exact_normalized_form_wins_a_key_collision():
    _, decisions = resolve_mentions([mention("Квантового  отжига")], REGISTRY)
    assert decisions[0].status == "accepted"
    assert decisions[0].concept_id == "concept:b"


def test_an_unresolved_collision_keeps_every_candidate():
    touched, decisions = resolve_mentions(
        [mention("квантовому отжигу")], REGISTRY
    )
    decision = decisions[0]
    assert decision.status == "ambiguous"
    assert decision.concept_id is None
    assert decision.candidates == [
        {"concept_id": "concept:a", "kind": "Technology", "score": 1.0},
        {"concept_id": "concept:b", "kind": "Technology", "score": 1.0},
    ]
    # Candidates are linked, not rewritten: their first_seen_at and names
    # must not change because of an unresolved mention.
    assert touched == []
