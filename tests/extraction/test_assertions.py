import pytest

from lctrend.core.models import (
    Chunk,
    Concept,
    ConceptKind,
    Mention,
    ResolutionDecision,
)
from lctrend.extraction.assertions import extract_assertions


def fixture(text, entities):
    chunk = Chunk(chunk_id="c", kind="abstract", text=text, order=0)
    concepts, mentions, decisions = [], [], []
    for index, (name, kind) in enumerate(entities):
        start = text.index(name)
        concept_id, mention_id = f"entity:{index}", f"mention:{index}"
        concepts.append(
            Concept(concept_id=concept_id, kind=kind, preferred_label=name)
        )
        mentions.append(
            Mention(
                mention_id=mention_id,
                chunk_id="c",
                surface_text=name,
                start=start,
                end=start + len(name),
                type_candidates=[kind],
            )
        )
        decisions.append(
            ResolutionDecision(
                resolution_id=f"resolution:{index}",
                mention_id=mention_id,
                status="provisional",
                concept_id=concept_id,
            )
        )
    return [chunk], mentions, concepts, decisions


@pytest.mark.parametrize(
    "connector,polarity,modality",
    [
        ("solves", "affirmed", "reported"),
        ("does not solve", "negated", "reported"),
        ("cannot solve", "negated", "reported"),
        ("may solve", "affirmed", "hypothetical"),
        ("can solve", "affirmed", "hypothetical"),
        ("will solve", "affirmed", "planned"),
    ],
)
def test_explicit_relations_are_candidates_with_literal_evidence(
    connector, polarity, modality
):
    data = fixture(
        f"  Technology A {connector} fraud detection.",
        [
            ("Technology A", ConceptKind.TECHNOLOGY),
            ("fraud detection", ConceptKind.TASK),
        ],
    )
    assertions = extract_assertions(*data)
    assert len(assertions) == 1
    item = assertions[0]
    assert item.predicate == "solves_task"
    assert item.roles == {"subject": "entity:0", "task": "entity:1"}
    assert item.polarity == polarity
    assert item.modality == modality
    assert item.status == "needs_review"
    assert item.verification_status == "unverified"
    assert item.extraction_confidence is None
    span = item.evidence[0]
    assert data[0][0].text[span.start : span.end] == span.quote


@pytest.mark.parametrize(
    "connector", ["for", "with", "to", "is related to", "is compared to"]
)
def test_prepositions_and_cooccurrence_do_not_prove_solves(connector):
    data = fixture(
        f"Technology A {connector} fraud detection.",
        [
            ("Technology A", ConceptKind.TECHNOLOGY),
            ("fraud detection", ConceptKind.TASK),
        ],
    )
    assert extract_assertions(*data) == []


def test_multiple_subjects_are_not_assigned_to_a_single_task():
    data = fixture(
        "Technology A and Technology B solve fraud detection.",
        [
            ("Technology A", ConceptKind.TECHNOLOGY),
            ("Technology B", ConceptKind.TECHNOLOGY),
            ("fraud detection", ConceptKind.TASK),
        ],
    )
    assert extract_assertions(*data) == []


def test_ambiguous_resolution_cannot_support_a_relation():
    data = fixture(
        "Technology A solves fraud detection.",
        [
            ("Technology A", ConceptKind.TECHNOLOGY),
            ("fraud detection", ConceptKind.TASK),
        ],
    )
    data[3][0].status = "ambiguous"
    assert extract_assertions(*data) == []


def test_explicit_metric_relation_keeps_negation():
    data = fixture(
        "Technology A does not reduce energy consumption.",
        [
            ("Technology A", ConceptKind.TECHNOLOGY),
            ("energy consumption", ConceptKind.METRIC),
        ],
    )
    items = extract_assertions(*data)
    assert len(items) == 1
    assert items[0].predicate == "changes_metric"
    assert items[0].polarity == "negated"


def test_lower_accuracy_is_a_change_not_an_assumed_advantage():
    data = fixture(
        "Technology A reduces accuracy.",
        [
            ("Technology A", ConceptKind.TECHNOLOGY),
            ("accuracy", ConceptKind.METRIC),
        ],
    )
    items = extract_assertions(*data)
    assert items[0].predicate == "changes_metric"
    assert items[0].qualifiers["relation_text"] == "reduces"
