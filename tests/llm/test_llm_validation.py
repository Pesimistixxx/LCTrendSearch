import pytest
from pydantic import ValidationError

from lctrend.core.models import (
    Artifact,
    Chunk,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.llm.contracts import (
    Extraction,
    LocalClaim,
    LocalEntity,
    Review,
    SourceSpan,
)
from lctrend.llm.validation import validate_local_extraction, validate_review


def document(text="Sensor S consumes 8 mW for monitoring."):
    return DocumentEnvelope(
        document_id="d",
        document_version_id="v",
        document_type=DocumentType.ARTICLE,
        title="Test",
        source=SourceRef(
            source_id="s", name="fixture", source_type="test", record_id="1"
        ),
        artifact=Artifact(
            uri="memory://1", sha256="a" * 64, media_type="text/plain"
        ),
        chunks=[
            Chunk(chunk_id="c1", kind="abstract", text=text, order=0),
            Chunk(
                chunk_id="c2",
                kind="paragraph",
                text="External definition.",
                order=1,
            ),
        ],
    )


def extraction(doc=None):
    doc = doc or document()
    return Extraction(
        entities=[
            LocalEntity(
                local_id="sensor",
                label="Sensor S",
                kind=ConceptKind.TECHNOLOGY,
                evidence=[SourceSpan(chunk_id="c1", quote="Sensor S")],
            ),
            LocalEntity(
                local_id="power",
                label="power consumption",
                kind=ConceptKind.METRIC,
                evidence=[SourceSpan(chunk_id="c1", quote="consumes")],
            ),
        ],
        claims=[
            LocalClaim(
                claim_id="a",
                predicate="reported_measurement",
                roles={"subject": "sensor", "metric": "power"},
                values=[
                    {
                        "entity_ref": "sensor",
                        "value": 8,
                        "unit": "mW",
                        "raw": "8 mW",
                    }
                ],
                polarity="affirmed",
                modality="observed",
                evidence=[SourceSpan(chunk_id="c1", quote=doc.chunks[0].text)],
            )
        ],
    )


def test_valid_payload_is_anchored_without_mutating_input_or_acceptance():
    doc = document()
    original = extraction(doc)
    validated, issues = validate_local_extraction(doc, original, ["c1"])
    assert issues == {}
    assert original.entities[0].evidence[0].start is None
    assert validated.entities[0].evidence[0].start == 0
    span = validated.claims[0].evidence[0]
    assert doc.chunks[0].text[span.start : span.end] == span.quote
    assert "status" not in validated.claims[0].model_dump()
    assert "verification_status" not in validated.claims[0].model_dump()


@pytest.mark.parametrize(
    "field", ["status", "verification_status", "concept_id"]
)
def test_llm_cannot_assign_acceptance_or_global_ids(field):
    payload = extraction().model_dump(mode="json")
    payload["claims"][0][field] = "accepted"
    with pytest.raises(ValidationError):
        Extraction.model_validate(payload)


def test_literal_evidence_cannot_reference_unseen_or_unknown_chunks():
    candidate = extraction()
    candidate.claims[0].evidence = [
        SourceSpan(chunk_id="c2", quote="External definition.")
    ]
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert "evidence_chunk_not_visible" in issues["claim:a"]
    candidate.claims[0].evidence[0].chunk_id = "absent"
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert "unknown_evidence_chunk" in issues["claim:a"]


def test_visible_ids_are_document_local():
    with pytest.raises(ValueError, match="unknown chunks"):
        validate_local_extraction(document(), extraction(), ["elsewhere"])


def test_rejected_source_chunk_cannot_support_visible_entities_or_claims():
    doc = document()
    doc.chunks[0].parse_status = "rejected"
    _, issues = validate_local_extraction(doc, extraction(doc), ["c1"])
    assert "evidence_chunk_parse_rejected" in issues["entity:sensor"]
    assert "evidence_chunk_parse_rejected" in issues["entity:power"]
    assert "evidence_chunk_parse_rejected" in issues["claim:a"]
    assert "invalid_entity:sensor" in issues["claim:a"]


@pytest.mark.parametrize("explicit_offsets", [False, True])
def test_generated_prefix_cannot_support_evidence_even_when_literal(
    explicit_offsets,
):
    prefix = "Generated heading.\n"
    doc = document(prefix + document().chunks[0].text)
    doc.chunks[0].locator["source_segments"] = [
        {
            "chunk_start": len(prefix),
            "chunk_end": len(doc.chunks[0].text),
            "source_start": 100,
            "source_end": 100 + len(doc.chunks[0].text) - len(prefix),
        }
    ]
    candidate = extraction(doc)
    candidate.entities[0].evidence = [
        SourceSpan(
            chunk_id="c1",
            quote="Generated heading.",
            start=0 if explicit_offsets else None,
            end=len("Generated heading.") if explicit_offsets else None,
        )
    ]
    _, issues = validate_local_extraction(doc, candidate, ["c1"])
    assert "quote_not_mapped_to_source" in issues["entity:sensor"]
    assert "invalid_entity:sensor" in issues["claim:a"]
    assert "quote_not_mapped_to_source" in issues["claim:a"]


def test_source_mapping_supports_partial_quotes_and_contiguous_source_slices():
    doc = document()
    split = 15
    doc.chunks[0].locator["source_segments"] = [
        {
            "chunk_start": split,
            "chunk_end": len(doc.chunks[0].text),
            "source_start": 100 + split,
            "source_end": 100 + len(doc.chunks[0].text),
        },
        {
            "chunk_start": 0,
            "chunk_end": split,
            "source_start": 100,
            "source_end": 100 + split,
        },
    ]
    anchored, issues = validate_local_extraction(doc, extraction(doc), ["c1"])
    assert issues == {}
    assert anchored.entities[0].evidence[0].start == 0
    assert anchored.claims[0].evidence[0].end == len(doc.chunks[0].text)


def test_quote_crossing_inserted_separator_is_not_source_evidence():
    first, last = "Sensor S", " consumes 8 mW for monitoring."
    doc = document(first + "\n" + last)
    doc.chunks[0].locator["source_segments"] = [
        {
            "chunk_start": 0,
            "chunk_end": len(first),
            "source_start": 100,
            "source_end": 100 + len(first),
        },
        {
            "chunk_start": len(first) + 1,
            "chunk_end": len(doc.chunks[0].text),
            "source_start": 100 + len(first),
            "source_end": 100 + len(first) + len(last),
        },
    ]
    _, issues = validate_local_extraction(doc, extraction(doc), ["c1"])
    assert "entity:sensor" not in issues
    assert "entity:power" not in issues
    assert issues["claim:a"] == ["quote_not_mapped_to_source"]


def test_adjacent_chunk_slices_with_discontinuous_source_cannot_support_one_quote():  # noqa: E501
    doc = document()
    split = 8
    doc.chunks[0].locator["source_segments"] = [
        {
            "chunk_start": 0,
            "chunk_end": split,
            "source_start": 100,
            "source_end": 100 + split,
        },
        {
            "chunk_start": split,
            "chunk_end": len(doc.chunks[0].text),
            "source_start": 200,
            "source_end": 200 + len(doc.chunks[0].text) - split,
        },
    ]
    _, issues = validate_local_extraction(doc, extraction(doc), ["c1"])
    assert "entity:sensor" not in issues
    assert "entity:power" not in issues
    assert issues["claim:a"] == ["noncontiguous_source_quote"]


def test_explicit_empty_source_mapping_does_not_fall_back_to_literal_text():
    doc = document()
    doc.chunks[0].locator["source_segments"] = []
    _, issues = validate_local_extraction(doc, extraction(doc), ["c1"])
    assert "quote_not_mapped_to_source" in issues["entity:sensor"]
    assert "quote_not_mapped_to_source" in issues["claim:a"]


@pytest.mark.parametrize(
    "segments",
    [
        None,
        [{"chunk_start": 0, "chunk_end": 8, "source_start": 100}],
        [
            {
                "chunk_start": False,
                "chunk_end": 8,
                "source_start": 100,
                "source_end": 108,
            }
        ],
        [
            {
                "chunk_start": 0,
                "chunk_end": 8,
                "source_start": 100,
                "source_end": 109,
            }
        ],
        [
            {
                "chunk_start": 0,
                "chunk_end": 99,
                "source_start": 100,
                "source_end": 199,
            }
        ],
        [
            {
                "chunk_start": 0,
                "chunk_end": 8,
                "source_start": 100,
                "source_end": 108,
            },
            {
                "chunk_start": 7,
                "chunk_end": 9,
                "source_start": 107,
                "source_end": 109,
            },
        ],
    ],
)
def test_malformed_or_overlapping_source_mapping_is_not_trusted(segments):
    doc = document()
    doc.chunks[0].locator["source_segments"] = segments
    _, issues = validate_local_extraction(doc, extraction(doc), ["c1"])
    assert "invalid_source_segments" in issues["entity:sensor"]
    assert "invalid_source_segments" in issues["claim:a"]


@pytest.mark.parametrize(
    "quote,start,end,expected",
    [
        ("invented", None, None, "quote_not_found"),
        ("Sensor S", -1, 7, "invalid_offsets"),
        ("Sensor S", 0, 99, "invalid_offsets"),
        ("Sensor S", 1, 9, "quote_offset_mismatch"),
        ("Sensor S", 0, None, "invalid_offsets"),
        (" ", None, None, "empty_quote"),
    ],
)
def test_invalid_quotes_and_offsets_have_explicit_gates(
    quote, start, end, expected
):
    candidate = extraction()
    candidate.entities[0].evidence = [
        SourceSpan(chunk_id="c1", quote=quote, start=start, end=end)
    ]
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert expected in issues["entity:sensor"]
    assert "invalid_entity:sensor" in issues["claim:a"]


def test_repeated_and_overlapping_quotes_need_explicit_offsets():
    doc = document("aaa Sensor S consumes 8 mW for monitoring.")
    candidate = extraction(doc)
    candidate.entities[0].evidence = [SourceSpan(chunk_id="c1", quote="aa")]
    _, issues = validate_local_extraction(doc, candidate, ["c1"])
    assert "ambiguous_quote" in issues["entity:sensor"]
    candidate.entities[0].evidence[0].start = 0
    candidate.entities[0].evidence[0].end = 2
    _, issues = validate_local_extraction(doc, candidate, ["c1"])
    assert issues == {}


def test_duplicate_entities_and_claims_invalidate_every_occurrence():
    candidate = extraction()
    candidate.entities.append(candidate.entities[0].model_copy(deep=True))
    candidate.claims.append(candidate.claims[0].model_copy(deep=True))
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert "duplicate_entity_id" in issues["entity:sensor"]
    assert "duplicate_claim_id" in issues["claim:a"]
    assert "invalid_entity:sensor" in issues["claim:a"]


@pytest.mark.parametrize(
    "change,expected",
    [
        (
            lambda claim: setattr(claim, "predicate", "invented_relation"),
            "unsupported_predicate",
        ),
        (lambda claim: claim.roles.pop("metric"), "missing_required_roles"),
        (
            lambda claim: claim.roles.update({"reporter": "sensor"}),
            "unsupported_role",
        ),
        (
            lambda claim: claim.roles.update({"subject": "power"}),
            "role_type_mismatch:subject",
        ),
        (
            lambda claim: claim.roles.update({"metric": "absent"}),
            "invalid_entity:absent",
        ),
    ],
)
def test_predicate_and_role_contracts_are_enforced(change, expected):
    candidate = extraction()
    change(candidate.claims[0])
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert expected in issues["claim:a"]


def test_recursive_entity_refs_in_qualifiers_and_values_are_validated():
    candidate = extraction()
    candidate.claims[0].qualifiers = {
        "group": [{"nested": {"entity_ref": "absent"}}]
    }
    candidate.claims[0].values[0]["context"] = {
        "nested": [{"entity_ref": ["sensor"]}]
    }
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert "invalid_entity_ref:absent" in issues["claim:a"]
    assert "invalid_entity_ref" in issues["claim:a"]


@pytest.mark.parametrize(
    "value,raw,expected",
    [
        (40, "8 mW", "number_not_in_raw_value"),
        (8, "8 watts", "value_not_grounded"),
        (float("nan"), "8 mW", "nonfinite_numeric_value"),
        (float("inf"), "8 mW", "nonfinite_numeric_value"),
        ("Infinity", "8 mW", "nonfinite_numeric_value"),
        ("NaN", "8 mW", "nonfinite_numeric_value"),
        (True, "8 mW", "invalid_numeric_value"),
    ],
)
def test_numeric_values_are_finite_and_literal(value, raw, expected):
    candidate = extraction()
    candidate.claims[0].values[0].update(value=value, raw=raw)
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert expected in issues["claim:a"]


def test_measurement_requires_reported_values():
    candidate = extraction()
    candidate.claims[0].values = []
    _, issues = validate_local_extraction(document(), candidate, ["c1"])
    assert "measurement_values_missing" in issues["claim:a"]


def test_signed_decimal_and_exponent_preserve_exact_numbers():
    for raw, value in [
        ("-8,5 mW", -8.5),
        ("1e-3 mW", 0.001),
        ("8mW", 8),
        (".5mW", 0.5),
    ]:
        doc = document("Sensor S consumes " + raw + " for monitoring.")
        candidate = extraction(doc)
        candidate.claims[0].values[0].update(value=value, raw=raw)
        _, issues = validate_local_extraction(doc, candidate, ["c1"])
        assert issues == {}


def test_review_must_cover_exactly_the_expected_ids_once():
    good = Review.model_validate(
        {
            "items": [
                {
                    "claim_id": "a",
                    "decision": "supported",
                    "reason": "Literal source statement.",
                }
            ]
        }
    )
    assert validate_review(good, ["a"]) == good
    for items in [
        [],
        [good.items[0], good.items[0]],
        [good.items[0].model_copy(update={"claim_id": "other"})],
    ]:
        with pytest.raises(ValueError):
            validate_review(Review(items=items), ["a"])
    with pytest.raises(ValueError):
        validate_review(good, ["a", "a"])


def test_review_decision_needs_a_reason_and_cannot_add_fields():
    with pytest.raises(ValueError):
        validate_review(
            Review.model_validate(
                {
                    "items": [
                        {
                            "claim_id": "a",
                            "decision": "supported",
                            "reason": " ",
                        }
                    ]
                }
            ),
            ["a"],
        )
    with pytest.raises(ValidationError):
        Review.model_validate(
            {
                "items": [
                    {
                        "claim_id": "a",
                        "decision": "supported",
                        "reason": "ok",
                        "status": "accepted",
                    }
                ]
            }
        )
