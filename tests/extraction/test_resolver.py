import sys
from types import SimpleNamespace

from lctrend.core.models import Concept, ConceptKind, ConceptName, Mention
from lctrend.extraction.resolver import (
    alias_keys,
    normalize_name,
    resolve_exact_mentions,
    resolve_mentions,
)


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
    new, decisions = resolve_exact_mentions(
        [mention("  НАТРИЙ-ИОННЫЙ   аккумулятор ")], [concept]
    )
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
    assert {name.text for name in concepts[0].names} == {
        "NLP",
        "natural language processing",
    }


def test_normalization_and_missing_optional_lemma(monkeypatch):
    assert normalize_name("  Graph-based_NER™ ") == "graph based ner"
    monkeypatch.setitem(sys.modules, "simplemma", None)
    assert not alias_keys("models") & alias_keys("model")
    assert alias_keys("  MODEL ") & alias_keys("model")


def test_optional_lemma_is_tested_with_injected_implementation(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "simplemma",
        SimpleNamespace(
            lemmatize=lambda token, lang: (
                "model" if token == "models" else token
            )
        ),
    )
    assert alias_keys("models") & alias_keys("model")


def test_same_initials_do_not_establish_identity():
    first = mention("carbon capture")
    second = mention("cloud computing")
    second.mention_id = "m2"
    concepts, decisions = resolve_exact_mentions([first, second], [])
    assert len(concepts) == 2
    assert decisions[0].concept_id != decisions[1].concept_id
    assert not alias_keys("carbon capture") & alias_keys("cloud computing")


def test_semantic_similarity_is_only_a_review_candidate():
    existing = Concept(
        concept_id="tech:one",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="carbon capture",
        status="accepted",
    )

    class FakeSemantic:
        def best_match(self, text, concepts):
            return existing, 0.99, 0.99

    touched, decisions = resolve_mentions(
        [mention("carbon conversion")], [existing], FakeSemantic()
    )
    assert touched == [existing]
    assert decisions[0].status == "ambiguous"
    assert decisions[0].concept_id is None
    assert decisions[0].review_status == "pending"
    assert decisions[0].candidates == [
        {"concept_id": "tech:one", "score": 0.99}
    ]
    assert existing.names == []


def test_unreviewed_observed_alias_does_not_merge_a_future_mention():
    existing = Concept(
        concept_id="tech:one",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="carbon capture",
        status="accepted",
        names=[
            ConceptName(
                name_id="observed",
                text="carbon conversion",
                normalized_text="carbon conversion",
                status="provisional",
            )
        ],
    )
    touched, decisions = resolve_exact_mentions(
        [mention("carbon conversion")], [existing]
    )
    assert touched[0].concept_id != existing.concept_id
    assert decisions[0].status == "provisional"
