import pytest

from lctrend.core.models import (
    Assertion,
    Chunk,
    Concept,
    ConceptKind,
    EvidenceSpan,
)
from lctrend.extraction.economics import (
    economic_evidence_from_assertions,
    extract_economic_evidence,
)
from lctrend.ingest.adapters import parse_openalex


def fixture():
    text = "Technology A reported a cost of EUR 2.5 million per year in 2025."
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": text}
    )
    document.chunks = [
        Chunk(chunk_id="c", kind="abstract", text=text, order=0)
    ]
    concepts = [
        Concept(
            concept_id="tech",
            kind=ConceptKind.TECHNOLOGY,
            preferred_label="Technology A",
        )
    ]
    assertion = Assertion(
        assertion_id="financial",
        predicate="reported_cost",
        roles={"subject": "tech"},
        evidence=[
            EvidenceSpan(chunk_id="c", quote=text, start=0, end=len(text))
        ],
        values=[
            {
                "raw": "EUR 2.5 million",
                "value": 2.5,
                "currency": "EUR",
                "unit": "per year",
                "period": "2025",
            }
        ],
        status="accepted",
        verification_status="supported",
    )
    return document, concepts, assertion


def test_reviewed_cost_retains_scale_period_and_assertion_provenance():
    document, concepts, assertion = fixture()
    result = economic_evidence_from_assertions(document, [assertion], concepts)
    assert len(result) == 1
    evidence = result[0]
    assert evidence.status == "accepted"
    assert evidence.assertion_id == "financial"
    assert evidence.amount_text == "EUR 2.5 million"
    assert evidence.amount_value == 2500000
    assert evidence.currency == "EUR"
    assert evidence.unit == "per year"
    assert evidence.period == "2025"
    assert (
        document.chunks[0].text[evidence.start : evidence.end]
        == evidence.quote
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "needs_review"},
        {"verification_status": "unsupported"},
        {"polarity": "negated"},
        {"modality": "planned"},
        {"modality": "hypothetical"},
        {"roles": {"subject": "missing"}},
    ],
)
def test_unreviewed_speculative_or_unattributed_finances_do_not_become_facts(
    changes,
):
    document, concepts, assertion = fixture()
    assertion = assertion.model_copy(update=changes)
    assert (
        economic_evidence_from_assertions(document, [assertion], concepts)
        == []
    )


def test_foreign_or_ungrounded_amount_is_not_primary_financial_evidence():
    document, concepts, assertion = fixture()
    assertion.values[0]["raw"] = "EUR 900 million"
    assert (
        economic_evidence_from_assertions(document, [assertion], concepts)
        == []
    )
    assertion.values[0]["raw"] = "EUR 2.5 million"
    assertion.evidence[0].chunk_id = "foreign"
    assert (
        economic_evidence_from_assertions(document, [assertion], concepts)
        == []
    )


def test_deduplicated_reviews_do_not_promote_regex_candidates():
    document, concepts, assertion = fixture()
    reviewed = economic_evidence_from_assertions(
        document, [assertion, assertion], concepts
    )
    assert len(reviewed) == 1
    assert extract_economic_evidence(document.chunks, [], concepts, []) == []


def test_bare_dollar_currency_is_retained_without_inventing_usd():
    document, concepts, assertion = fixture()
    text = "Technology A reported a cost of $2.5 million."
    document.chunks[0].text = text
    assertion.evidence = [
        EvidenceSpan(chunk_id="c", quote=text, start=0, end=len(text))
    ]
    assertion.values = [{"raw": "$2.5 million", "value": 2.5, "currency": "$"}]
    result = economic_evidence_from_assertions(document, [assertion], concepts)
    assert result[0].currency == "$"
    assert result[0].period is None
