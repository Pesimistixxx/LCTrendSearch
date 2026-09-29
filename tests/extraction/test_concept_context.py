"""Kinds settled by votes, declared aliases and definitions of concepts."""

from itertools import permutations

from lctrend.core.models import (
    DECLARED_ALIAS_METHOD,
    Concept,
    ConceptKind,
    ConceptName,
    Mention,
)
from lctrend.extraction.lexical import settled_kind
from lctrend.extraction.resolver import (
    ConceptIndex,
    concept_text,
    context_text,
    observed_name,
    resolve_mentions,
)

T = ConceptKind.TECHNOLOGY
MATERIAL = ConceptKind.MATERIAL


def mention(mention_id, text, kind=T, surface=None, **fields):
    surface = surface or text
    return Mention(
        mention_id=mention_id,
        chunk_id="c1",
        surface_text=surface,
        canonical_text=text,
        start=0,
        end=len(surface),
        type_candidates=[kind],
        **fields,
    )


def resolve(index, *mentions):
    return resolve_mentions(list(mentions), index)


def test_a_technology_family_kind_is_the_most_reported_one():
    assert settled_kind({"Technology": 1, "Material": 3}, T) == "Material"
    # A tie goes to the higher rank, whatever came first.
    assert settled_kind({"Material": 2, "Technology": 2}, T) == "Technology"
    # Kinds of another family do not vote.
    assert settled_kind({"Company": 9, "Material": 1}, T) == "Material"


def test_an_organization_takes_its_highest_kind():
    counts = {"Organization": 5, "Company": 1}
    assert settled_kind(counts, ConceptKind.ORGANIZATION) == "Company"


def test_one_technology_mention_does_not_relabel_a_compound_for_good():
    kinds = [MATERIAL, T, MATERIAL, MATERIAL]
    results = set()
    for order in permutations(range(len(kinds))):
        index = ConceptIndex()
        for position in order:
            resolve(index, mention(f"m{position}", "ML-236B", kinds[position]))
        (concept,) = list(index)
        results.add((concept.kind, tuple(sorted(concept.kind_counts.items()))))
    assert results == {(MATERIAL, (("Material", 3), ("Technology", 1)))}


def test_a_stored_concept_without_votes_counts_its_mentions_as_its_kind():
    stored = Concept(
        concept_id="concept:ml",
        kind=T,
        preferred_label="ML-236B",
        identity_key="ml 236 b",
        label_counts={"ML-236B": 5},
    )
    index = ConceptIndex([stored])
    resolve(index, mention("m1", "ML-236B", MATERIAL))
    concept = index.get("concept:ml")
    assert concept.kind_counts == {"Technology": 5, "Material": 1}
    assert concept.kind == T


def test_a_quote_is_recorded_as_the_label_not_as_a_name():
    quote = "ML-236A, ML-236B and ML-236C, new inhibitors of cholesterogenesis"
    assert observed_name(quote, "ML-236B") == "ML-236B"
    assert observed_name("LLMs", "large language models") == "LLMs"
    assert observed_name("Германии", "DE") == "Германии"
    index = ConceptIndex()
    resolve(index, mention("m1", "ML-236B", MATERIAL, surface=quote))
    (concept,) = list(index)
    assert [name.text for name in concept.names] == ["ML-236B"]


def test_a_declared_alias_names_the_concept_in_later_documents():
    index = ConceptIndex()
    resolve(
        index,
        mention("m1", "compactin", MATERIAL, declared_aliases=["ML-236B"]),
    )
    concepts, decisions = resolve(index, mention("m2", "ML-236B", MATERIAL))
    (concept,) = list(index)
    assert decisions[0].concept_id == concept.concept_id
    assert decisions[0].status == "accepted"
    declared = [name for name in concept.names if name.name_kind == "declared"]
    assert [(name.text, name.status) for name in declared] == [
        ("ML-236B", "accepted")
    ]


def test_a_declared_alias_resolves_a_label_the_registry_does_not_know():
    index = ConceptIndex()
    resolve(index, mention("m1", "ML-236B", MATERIAL))
    _, decisions = resolve(
        index,
        mention("m2", "compactin", MATERIAL, declared_aliases=["ML-236B"]),
    )
    (concept,) = list(index)
    assert decisions[0].method == DECLARED_ALIAS_METHOD
    assert decisions[0].concept_id == concept.concept_id
    # The label now names the concept too: "compactin" alone resolves.
    _, decisions = resolve(index, mention("m3", "compactin", MATERIAL))
    assert decisions[0].concept_id == concept.concept_id
    assert len(index) == 1


def test_a_declared_alias_of_another_concept_becomes_a_merge_candidate():
    index = ConceptIndex()
    resolve(
        index,
        mention("m1", "ML-236B", MATERIAL),
        mention("m2", "compactin", MATERIAL),
    )
    by_label = {concept.preferred_label: concept for concept in index}
    _, decisions = resolve(
        index,
        mention("m3", "compactin", MATERIAL, declared_aliases=["ML-236B"]),
    )
    (decision,) = decisions
    assert decision.concept_id == by_label["compactin"].concept_id
    assert decision.candidates == [
        {
            "concept_id": by_label["ML-236B"].concept_id,
            "kind": "Material",
            "score": 1.0,
            "method": DECLARED_ALIAS_METHOD,
            "alias": "ML-236B",
        }
    ]
    # The alias is not taken over: "ML-236B" still names one concept.
    _, decisions = resolve(index, mention("m4", "ML-236B", MATERIAL))
    assert decisions[0].concept_id == by_label["ML-236B"].concept_id


def test_a_declared_alias_of_another_family_is_ignored():
    index = ConceptIndex()
    resolve(index, mention("m1", "Merck", ConceptKind.COMPANY))
    _, decisions = resolve(
        index,
        mention("m2", "lovastatin", MATERIAL, declared_aliases=["Merck"]),
    )
    assert decisions[0].candidates == []
    lovastatin = next(c for c in index if c.preferred_label == "lovastatin")
    assert "Merck" not in [name.text for name in lovastatin.names]


def test_a_concept_keeps_the_first_definition_a_source_gives():
    index = ConceptIndex()
    resolve(
        index,
        mention(
            "m1",
            "ML-236B",
            MATERIAL,
            definition="inhibitor of cholesterol synthesis",
        ),
        mention("m2", "ML-236B", MATERIAL, definition="a fungal metabolite"),
    )
    (concept,) = list(index)
    assert concept.definition == "inhibitor of cholesterol synthesis"
    assert concept_text(concept) == (
        "ML-236B: inhibitor of cholesterol synthesis"
    )
    assert context_text("ML-236B") == "ML-236B"
    assert context_text("X", "  a   b ") == "X: a b"


def test_semantic_matching_compares_names_with_their_definitions():
    seen = []

    class Semantic:
        def available(self):
            return True

        def embed(self, texts):
            seen.append(("embed", list(texts)))

        def best_match(self, text, concepts):
            seen.append(("match", text, [concept_text(c) for c in concepts]))
            return None, 0.0, 0.0

    stored = Concept(
        concept_id="concept:c",
        kind=MATERIAL,
        preferred_label="compactin",
        definition="HMG-CoA reductase inhibitor",
        names=[
            ConceptName(
                name_id="n", text="compactin", normalized_text="compactin"
            )
        ],
    )
    resolve_mentions(
        [
            mention(
                "m1", "ML-236B", MATERIAL, definition="cholesterol inhibitor"
            )
        ],
        ConceptIndex([stored]),
        Semantic(),
    )
    assert ("match", "ML-236B: cholesterol inhibitor",
            ["compactin: HMG-CoA reductase inhibitor"]) in seen
