from lctrend.economics import extract_economic_evidence
from lctrend.models import (
    Chunk,
    Concept,
    ConceptKind,
    Mention,
    ResolutionDecision,
)


def test_economic_evidence_requires_economic_text_and_technology_in_same_sentence():
    chunks = [
        Chunk(
            chunk_id="ch1",
            kind="abstract",
            text="GLiNER reduces inference cost by 40%. It is easy to install.",
            order=0,
        )
    ]
    concepts = [
        Concept(concept_id="tech1", kind=ConceptKind.TECHNOLOGY, preferred_label="GLiNER")
    ]
    mentions = [
        Mention(
            mention_id="m1",
            chunk_id="ch1",
            surface_text="GLiNER",
            start=0,
            end=6,
            type_candidates=[ConceptKind.TECHNOLOGY],
        )
    ]
    resolutions = [
        ResolutionDecision(
            resolution_id="r1",
            mention_id="m1",
            status="accepted",
            concept_id="tech1",
        )
    ]

    evidence = extract_economic_evidence(chunks, mentions, concepts, resolutions)

    assert len(evidence) == 1
    assert evidence[0].technology_concept_id == "tech1"
    assert evidence[0].category == "cost"
    assert evidence[0].quote == "GLiNER reduces inference cost by 40%."


def test_economic_evidence_is_not_inferred_without_explicit_economic_language():
    chunk = Chunk(
        chunk_id="ch1", kind="abstract", text="GLiNER extracts named entities.", order=0
    )
    concept = Concept(
        concept_id="tech1", kind=ConceptKind.TECHNOLOGY, preferred_label="GLiNER"
    )
    mention = Mention(
        mention_id="m1",
        chunk_id="ch1",
        surface_text="GLiNER",
        start=0,
        end=6,
        type_candidates=[ConceptKind.TECHNOLOGY],
    )
    resolution = ResolutionDecision(
        resolution_id="r1", mention_id="m1", status="accepted", concept_id="tech1"
    )

    assert extract_economic_evidence([chunk], [mention], [concept], [resolution]) == []
