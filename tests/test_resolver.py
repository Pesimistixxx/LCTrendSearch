from lctrend.models import Concept, ConceptKind, ConceptName, Mention
from lctrend.resolver import alias_keys, normalize_name, resolve_exact_mentions


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
    assert new == [concept]
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


def test_repeated_exact_mentions_share_one_provisional_concept():
    first = mention("GLiNER")
    second = mention("gliner")
    second.mention_id = "m2"
    new, decisions = resolve_exact_mentions([first, second], [])
    assert len(new) == 1
    assert decisions[0].concept_id == decisions[1].concept_id


def test_abbreviation_and_full_name_share_one_concept():
    first = mention("NLP")
    second = mention("natural language processing")
    second.mention_id = "m2"
    concepts, decisions = resolve_exact_mentions([first, second], [])
    assert len(concepts) == 1
    assert decisions[0].concept_id == decisions[1].concept_id
    assert {name.text for name in concepts[0].names} == {"NLP", "natural language processing"}


def test_normalization_removes_symbols_and_lemmatizes():
    assert normalize_name("  Graph-based_NER™ ") == "graph based ner"
    assert alias_keys("models") & alias_keys("model")
