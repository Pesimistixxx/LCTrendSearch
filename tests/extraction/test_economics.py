from lctrend.core.models import (
    Chunk,
    Concept,
    ConceptKind,
    Mention,
    ResolutionDecision,
)
from lctrend.extraction.economics import extract_economic_evidence


def test_economic_evidence_requires_economic_text_and_technology_in_same_sentence():  # noqa: E501
    chunks = [
        Chunk(
            chunk_id="ch1",
            kind="abstract",
            text=(
                "GLiNER reduces operating cost by 40%. It is easy to install."
            ),
            order=0,
        )
    ]
    concepts = [
        Concept(
            concept_id="tech1",
            kind=ConceptKind.TECHNOLOGY,
            preferred_label="GLiNER",
        )
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

    evidence = extract_economic_evidence(
        chunks, mentions, concepts, resolutions
    )

    assert len(evidence) == 1
    assert evidence[0].technology_concept_id == "tech1"
    assert evidence[0].category == "cost"
    assert evidence[0].quote == "GLiNER reduces operating cost by 40%."


def test_economic_evidence_is_not_inferred_without_explicit_economic_language():  # noqa: E501
    chunk = Chunk(
        chunk_id="ch1",
        kind="abstract",
        text="GLiNER extracts named entities.",
        order=0,
    )
    concept = Concept(
        concept_id="tech1",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="GLiNER",
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
        resolution_id="r1",
        mention_id="m1",
        status="accepted",
        concept_id="tech1",
    )

    assert (
        extract_economic_evidence([chunk], [mention], [concept], [resolution])
        == []
    )


def economic_fixture(text, names=("Technology A",)):
    chunk = Chunk(chunk_id="c", kind="abstract", text=text, order=0)
    concepts, mentions, resolutions = [], [], []
    for index, name in enumerate(names):
        start = text.index(name)
        concepts.append(
            Concept(
                concept_id=f"tech:{index}",
                kind=ConceptKind.TECHNOLOGY,
                preferred_label=name,
            )
        )
        mentions.append(
            Mention(
                mention_id=f"m:{index}",
                chunk_id="c",
                surface_text=name,
                start=start,
                end=start + len(name),
                type_candidates=[ConceptKind.TECHNOLOGY],
            )
        )
        resolutions.append(
            ResolutionDecision(
                resolution_id=f"r:{index}",
                mention_id=f"m:{index}",
                status="accepted",
                concept_id=f"tech:{index}",
            )
        )
    return [chunk], mentions, concepts, resolutions


def test_computational_and_inference_cost_are_not_monetary():
    for cost in (
        "computational cost",
        "inference cost",
        "time cost",
        "вычислительные затраты",
    ):
        data = economic_fixture(f"Technology A reduces {cost} by 40%.")
        assert extract_economic_evidence(*data) == []


def test_negated_savings_and_planned_cost_are_not_facts():
    data = economic_fixture(
        "Technology A may not reduce operating cost by $10."
    )
    evidence = extract_economic_evidence(*data)
    assert len(evidence) == 1
    assert evidence[0].polarity == "negated"
    assert evidence[0].modality == "hypothetical"
    assert evidence[0].status == "candidate"
    assert evidence[0].confidence is None
    assert evidence[0].amount_text == "$10"
    assert evidence[0].currency == "USD"


def test_separate_clause_amounts_belong_to_separate_technologies():
    data = economic_fixture(
        "Technology A costs $10; Technology B costs $20.",
        ("Technology A", "Technology B"),
    )
    evidence = extract_economic_evidence(*data)
    assert [
        (item.technology_concept_id, item.amount_text) for item in evidence
    ] == [("tech:0", "$10"), ("tech:1", "$20")]
    assert all(
        data[0][0].text[item.start : item.end] == item.quote
        for item in evidence
    )


def test_multiple_technology_owners_are_skipped_instead_of_copying_amount():
    data = economic_fixture(
        "Technology A and Technology B cost $10.",
        ("Technology A", "Technology B"),
    )
    assert extract_economic_evidence(*data) == []


def test_ambiguous_resolution_is_not_economic_ownership():
    data = economic_fixture("Technology A costs $10.")
    data[3][0].status = "ambiguous"
    assert extract_economic_evidence(*data) == []


def test_decimal_money_stays_in_one_evidence_span():
    data = economic_fixture("Technology A costs $10.25.")
    items = extract_economic_evidence(*data)
    assert len(items) == 1
    assert items[0].amount_text == "$10.25"
    assert items[0].quote == data[0][0].text


def test_cooccurring_company_investment_is_not_assigned_to_technology():
    data = economic_fixture(
        "Unlike Technology A, Company B received $10 million in funding."
    )
    assert extract_economic_evidence(*data) == []


def test_explicit_cost_of_technology_is_kept():
    data = economic_fixture("The cost of Technology A is $10.")
    items = extract_economic_evidence(*data)
    assert len(items) == 1
    assert items[0].amount_text == "$10"


def test_amounts_are_read_with_scale_words_and_ambiguity_is_not_guessed():
    from lctrend.core.config import load_catalog
    from lctrend.extraction.economics import amount_value

    scales = load_catalog("extraction")["economics"]["scales"]
    assert amount_value("$5 million", scales) == 5e6
    assert amount_value("1,5 млрд руб", scales) == 1.5e9
    assert amount_value("1 000 000 USD", scales) == 1e6
    assert amount_value("$1,200,000", scales) == 1.2e6
    assert amount_value("1.2.3 EUR", scales) is None
