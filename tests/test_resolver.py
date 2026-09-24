from lctrend.models import Concept, ConceptKind, ConceptName, Mention
from lctrend.resolver import normalize_name, resolve_exact_mentions


def mention(text="Натрий-ионный аккумулятор"):
    return Mention(
        mention_id="m1",
        chunk_id="c1",
        surface_text=text,
        start=0,
        end=len(text),
        type_candidates=[ConceptKind.TECHNOLOGY],
    )


def test_reviewed_exact_name_is_resolved():
    concept = Concept(
        concept_id="tech:sodium",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="Натрий-ионный аккумулятор",
        status="accepted",
        names=[
            ConceptName(
                name_id="n1",
                text="Натрий-ионный аккумулятор",
                normalized_text=normalize_name("Натрий-ионный аккумулятор"),
            )
        ],
    )
    new, decisions = resolve_exact_mentions([mention("  НАТРИЙ-ИОННЫЙ   аккумулятор ")], [concept])
    assert not new
    assert decisions[0].concept_id == "tech:sodium"
    assert decisions[0].status == "accepted"


def test_similar_but_distinct_technology_is_not_merged():
    lithium = Concept(
        concept_id="tech:lithium",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="Литий-ионный аккумулятор",
        status="accepted",
    )
    new, decisions = resolve_exact_mentions([mention()], [lithium])
    assert new[0].preferred_label == "Натрий-ионный аккумулятор"
    assert new[0].status == "provisional"
    assert decisions[0].concept_id != "tech:lithium"
