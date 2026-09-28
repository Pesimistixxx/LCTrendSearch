"""Offline end-to-end checks for packet processing, not model quality
benchmarks.
"""

import asyncio
import json
from copy import deepcopy

import pytest

from lctrend.core.models import (
    Artifact,
    Chunk,
    Concept,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.graph.store import GraphStore
from lctrend.llm.client import LLMError, ReplayProvider
from lctrend.llm.context import PipelineSettings, build_payload, plan_packets
from lctrend.llm.contracts import Extraction
from lctrend.llm.pipeline import process_document

pytestmark = pytest.mark.legacy_technology_entities


def document(texts=None, shared_stream=False):
    texts = texts or ["Sensor S solves monitoring."]
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="version",
        document_type=DocumentType.REPORT,
        title="Sensor study",
        source=SourceRef(
            source_id="fixture",
            name="fixture",
            source_type="test",
            record_id="1",
        ),
        artifact=Artifact(
            uri="memory://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        chunks=[
            Chunk(
                chunk_id=f"c{index + 1}",
                kind="paragraph",
                text=text,
                order=index,
                locator={
                    "path": "study.md" if shared_stream else f"part-{index}.md"
                },
            )
            for index, text in enumerate(texts)
        ],
    )


def settings(**overrides):
    values = {
        "primary_chunks": 1,
        "max_context_chunks": 4,
        "max_source_chars": 20000,
        "max_payload_chars": 100000,
        "max_model_calls": 8,
        "max_context_rounds": 2,
        "max_retries": 0,
        "retry_delay_seconds": 0,
        "max_retry_delay_seconds": 0,
    }
    values.update(overrides)
    return PipelineSettings(**values)


def extracted(
    doc,
    chunk_id="c1",
    *,
    polarity="affirmed",
    modality="reported",
    claim_id="claim",
):
    chunk = next(chunk for chunk in doc.chunks if chunk.chunk_id == chunk_id)
    return {
        "entities": [
            {
                "local_id": "sensor",
                "label": "Sensor S",
                "kind": "Method",
                "evidence": [{"chunk_id": chunk_id, "quote": "Sensor S"}],
            },
            {
                "local_id": "task",
                "label": "monitoring",
                "kind": "Task",
                "evidence": [{"chunk_id": chunk_id, "quote": "monitoring"}],
            },
        ],
        "claims": [
            {
                "claim_id": claim_id,
                "predicate": "solves_task",
                "roles": {"subject": "sensor", "task": "task"},
                "qualifiers": {},
                "values": [],
                "polarity": polarity,
                "modality": modality,
                "attribution_kind": "author_reported",
                "evidence": [{"chunk_id": chunk_id, "quote": chunk.text}],
            }
        ],
        "context_requests": [],
    }


def reviewed(decision="supported", claim_id="claim"):
    return {
        "items": [
            {
                "claim_id": claim_id,
                "decision": decision,
                "reason": "Source text checked.",
            }
        ]
    }


class RecordingReplay(ReplayProvider):
    def __init__(self, answers):
        super().__init__(answers)
        self.payloads = []

    async def generate(self, schema, system, payload, *, stage="extract"):
        self.payloads.append({"stage": stage, "payload": deepcopy(payload)})
        return await super().generate(schema, system, payload, stage=stage)


class ScriptProvider:
    """Count failed attempts too, without sending network traffic or
    sleeping.
    """

    demo = False
    models = {"extract": "offline-extractor", "review": "offline-reviewer"}

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate(self, schema, system, payload, *, stage="extract"):
        self.calls.append({"stage": stage, "schema": schema.__name__})
        if not self.answers:
            raise AssertionError("Unexpected provider call")
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return schema.model_validate(answer)


def codes(result):
    return {
        item.get("code")
        for item in result.run.metadata["issues"]
        if "code" in item
    }


def test_supported_source_claim_is_accepted_with_exact_anchors_and_local_roles():  # noqa: E501
    doc = document()
    provider = RecordingReplay([extracted(doc), reviewed()])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert len(result.assertions) == 1
    claim = result.assertions[0]
    assert claim.status == "accepted"
    assert claim.verification_status == "supported"
    assert set(claim.roles.values()).issubset(
        {concept.concept_id for concept in result.concepts}
    )
    assert all(
        doc.chunks[0].text[span.start : span.end] == span.quote
        for span in claim.evidence
    )
    assert result.run.metadata["source_truth_assessed"] is False
    assert result.run.metadata["demo"] is True
    assert result.run.metadata["model_calls"] == 2
    assert [item["stage"] for item in provider.payloads] == [
        "extract",
        "review",
    ]
    assert result.run.status == "succeeded"


def test_invalid_quote_is_not_sent_to_reviewer_and_has_an_audit_gate():
    doc = document()
    response = extracted(doc)
    response["claims"][0]["evidence"][0]["quote"] = "Invented source sentence."
    provider = RecordingReplay([response, {"items": []}])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert result.assertions == []
    assert len(provider.calls) == 2
    assert any(
        item.get("item") == "claim:claim"
        and "quote_not_found" in item["reasons"]
        for item in result.run.metadata["issues"]
    )
    # The gated claim is audited; the chunk itself was fully processed.
    assert result.run.status == "succeeded"
    assert result.run.metadata["item_issue_count"] == 1


def test_context_loop_adds_original_chunk_and_replaces_previous_answer():
    doc = document(
        [
            "Sensor S solves monitoring.",
            "Background only.",
            "Definition of the sensing system.",
        ]
    )
    first = {
        "entities": [],
        "claims": [],
        "context_requests": [
            {
                "tool": "read_chunk",
                "argument": "c3",
                "reason": "Read the definition.",
            }
        ],
    }
    provider = RecordingReplay([first, extracted(doc), reviewed()])
    result = asyncio.run(
        process_document(doc, provider, settings=settings(max_model_calls=3))
    )
    assert [item["stage"] for item in provider.payloads] == [
        "extract",
        "extract",
        "review",
    ]
    assert {
        item["chunk_id"] for item in provider.payloads[0]["payload"]["chunks"]
    } == {"c1"}
    assert {
        item["chunk_id"] for item in provider.payloads[1]["payload"]["chunks"]
    } == {"c1", "c3"}
    assert result.assertions[0].status == "accepted"
    assert any(
        item["stage"] == "context"
        and item["outcomes"][0]["added_chunk_ids"] == ["c3"]
        for item in result.run.trace
    )
    assert "unresolved_context" not in codes(result)
    assert result.run.metadata["coverage"]["unprocessed_chunk_ids"] == [
        "c2",
        "c3",
    ]


def test_context_budget_preserves_the_reviewer_call_and_blocks_acceptance():
    doc = document(["Sensor S solves monitoring.", "Required definition."])
    first = extracted(doc)
    first["context_requests"] = [
        {
            "tool": "read_chunk",
            "argument": "c2",
            "reason": "Missing condition.",
        }
    ]
    provider = RecordingReplay([first, reviewed()])
    result = asyncio.run(
        process_document(doc, provider, settings=settings(max_model_calls=2))
    )
    assert [call["stage"] for call in provider.calls] == ["extract", "review"]
    assert result.run.metadata["model_calls"] == 2
    assert result.assertions[0].status == "needs_review"
    assert result.assertions[0].verification_status == "unverified"
    assert "unresolved_context" in codes(result)


@pytest.mark.parametrize("max_calls", [3, 2])
def test_already_visible_context_request_reextracts_only_with_reviewer_reserve(
    max_calls,
):
    doc = document()
    first = extracted(doc)
    first["context_requests"] = [
        {
            "tool": "read_chunk",
            "argument": "c1",
            "reason": "Re-read the original condition.",
        }
    ]
    answers = (
        [
            first,
            extracted(doc, claim_id="replacement"),
            reviewed(claim_id="replacement"),
        ]
        if max_calls == 3
        else [first, reviewed()]
    )
    provider = RecordingReplay(answers)
    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(max_model_calls=max_calls)
        )
    )

    assert result.run.metadata["model_calls"] == max_calls
    assert len(result.assertions) == 1
    claim = result.assertions[0]
    if max_calls == 3:
        assert [item["stage"] for item in provider.payloads] == [
            "extract",
            "extract",
            "review",
        ]
        assert (
            provider.payloads[1]["payload"]["chunks"]
            == provider.payloads[0]["payload"]["chunks"]
        )
        assert "already_visible" in str(
            provider.payloads[1]["payload"]["feedback"]
        )
        assert (
            provider.payloads[2]["payload"]["extraction"]["claims"][0][
                "claim_id"
            ]
            == "replacement"
        )
        assert any(
            item["stage"] == "context"
            and item["outcomes"][0]["status"] == "already_visible"
            for item in result.run.trace
        )
        assert claim.status == "accepted"
        assert claim.verification_status == "supported"
        assert "unresolved_context" not in codes(result)
        assert result.run.status == "succeeded"
    else:
        assert [item["stage"] for item in provider.payloads] == [
            "extract",
            "review",
        ]
        assert claim.status == "needs_review"
        assert claim.verification_status == "unverified"
        assert "unresolved_context" in codes(result)
        assert result.run.status == "partial"


@pytest.mark.parametrize(
    "decision,status,verification",
    [
        ("unsupported", "rejected", "unsupported"),
        ("unclear", "needs_review", "unverified"),
    ],
)
def test_non_supporting_review_never_becomes_accepted(
    decision, status, verification
):
    doc = document()
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider([extracted(doc), reviewed(decision)]),
            settings=settings(),
        )
    )
    assert result.assertions[0].status == status
    assert result.assertions[0].verification_status == verification
    assert GraphStore._solution_links(doc, result) == []
    if decision == "unclear":
        # Stored as needs_review and kept out of projections; the document
        # is still completely processed.
        assert result.run.status == "succeeded"
        assert "review_unclear" in codes(result)
    else:
        # A definite rejection completes review; it is not an incomplete run.
        assert result.run.status == "succeeded"


def test_reviewer_failure_preserves_anchored_candidate_and_records_failure():
    doc = document()
    provider = ScriptProvider(
        [extracted(doc), LLMError("timeout", "Offline timeout")]
    )
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert result.assertions[0].status == "needs_review"
    assert result.assertions[0].verification_status == "unverified"
    assert "timeout" in codes(result)
    assert result.run.status == "partial"
    assert result.run.metadata["coverage"]["processed_focus_chunk_ids"] == [
        "c1"
    ]


@pytest.mark.parametrize(
    "bad_review",
    [
        {"items": []},
        {"items": reviewed()["items"] * 2},
        reviewed(claim_id="foreign"),
    ],
)
def test_bad_reviewer_id_contract_cannot_accept_claim(bad_review):
    doc = document()
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider([extracted(doc), bad_review]),
            settings=settings(),
        )
    )
    assert result.assertions[0].status == "needs_review"
    assert "review_contract" in codes(result)


def test_retryable_failure_counts_against_the_document_budget():
    doc = document()
    provider = ScriptProvider(
        [
            LLMError("timeout", "Retry once", retryable=True),
            extracted(doc),
            reviewed(),
        ]
    )
    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(max_model_calls=3, max_retries=1)
        )
    )
    assert len(provider.calls) == 3
    assert result.run.metadata["model_calls"] == 3
    assert result.assertions[0].status == "accepted"
    failures = [
        item for item in result.run.trace if item.get("status") == "failed"
    ]
    assert len(failures) == 1 and failures[0]["call"] == 1


def test_retry_limit_stops_before_a_third_attempt():
    doc = document()
    provider = ScriptProvider(
        [
            LLMError("timeout", "One", retryable=True),
            LLMError("timeout", "Two", retryable=True),
            extracted(doc),
        ]
    )
    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(max_model_calls=8, max_retries=1)
        )
    )
    assert len(provider.calls) == 2
    assert result.run.metadata["model_calls"] == 2
    assert result.run.status == "failed"
    assert result.assertions == []
    assert result.run.metadata["coverage"]["unprocessed_chunk_ids"] == ["c1"]


def test_retry_cannot_consume_reserved_reviewer_budget():
    doc = document()
    provider = ScriptProvider(
        [
            LLMError("timeout", "One", retryable=True),
            extracted(doc),
            reviewed(),
        ]
    )
    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(max_model_calls=2, max_retries=4)
        )
    )
    assert len(provider.calls) == 1
    assert result.run.metadata["model_calls"] == 1
    assert result.run.status == "failed"


def test_nonretryable_invalid_schema_does_not_retry():
    doc = document()
    provider = ScriptProvider(
        [{"entities": [], "claims": [], "unexpected": True}, extracted(doc)]
    )
    result = asyncio.run(
        process_document(doc, provider, settings=settings(max_retries=4))
    )
    assert len(provider.calls) == 1
    assert result.run.status == "failed"
    assert "invalid_response" in codes(result)


def test_omitted_and_unvisited_chunks_are_explicit_and_successful_packet_is_kept():  # noqa: E501
    doc = document(
        ["Sensor S solves monitoring.", "X" * 1000, "A third valid chunk."]
    )
    provider = ReplayProvider([extracted(doc), reviewed()])
    result = asyncio.run(
        process_document(
            doc,
            provider,
            settings=settings(max_model_calls=2, max_source_chars=100),
        )
    )
    assert result.assertions[0].status == "accepted"
    assert result.run.status == "partial"
    coverage = result.run.metadata["coverage"]
    assert coverage["omitted_chunk_ids"] == ["c2"]
    assert coverage["unprocessed_chunk_ids"] == ["c2", "c3"]
    assert coverage["processed_focus_chunk_ids"] == ["c1"]
    assert result.run.metadata["model_calls"] == 2


def test_ambiguous_registry_identity_keeps_source_claim_out_of_compiled_assertions():  # noqa: E501
    doc = document()
    registry = [
        Concept(
            concept_id=f"existing:{index}",
            preferred_label="Sensor S",
            kind=ConceptKind.METHOD,
            status="accepted",
        )
        for index in range(2)
    ]
    original = deepcopy(registry)
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider([extracted(doc), reviewed()]),
            registry,
            settings(),
        )
    )
    assert result.assertions == []
    assert any(item.status == "ambiguous" for item in result.resolutions)
    assert result.run.metadata["unresolved_claims"][0]["review"] == "supported"
    assert (
        result.run.metadata["unresolved_claims"][0]["claim"]["evidence"][0][
            "quote"
        ]
        == doc.chunks[0].text
    )
    assert result.run.status == "succeeded"
    assert registry == original


def test_negation_and_planned_modality_survive_supported_text_review():
    doc = document(["Sensor S will not solve monitoring."])
    response = extracted(doc, polarity="negated", modality="planned")
    result = asyncio.run(
        process_document(
            doc, ReplayProvider([response, reviewed()]), settings=settings()
        )
    )
    claim = result.assertions[0]
    assert (
        claim.status == "accepted"
    )  # Accepted as a source statement, not a proven solution.
    assert claim.polarity == "negated"
    assert claim.modality == "planned"
    assert GraphStore._solution_links(doc, result) == []


def test_exact_claim_repeated_in_overlapping_context_is_one_assertion():
    doc = document(
        [
            "Opening context.",
            "Sensor S solves monitoring.",
            "Closing context.",
        ],
        shared_stream=True,
    )
    response = extracted(doc, "c2")
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider(
                [
                    response,
                    reviewed(),
                    response,
                    reviewed(),
                    response,
                    reviewed(),
                ]
            ),
            settings=settings(max_model_calls=6),
        )
    )
    assert len(result.assertions) == 1
    assert len(result.mentions) == 2
    assert result.run.metadata["model_calls"] == 6
    assert result.run.metadata["coverage"]["processed_focus_chunk_ids"] == [
        "c1",
        "c2",
        "c3",
    ]


def test_conflicting_reviews_for_same_source_claim_cannot_restore_acceptance():
    doc = document(
        [
            "Opening context.",
            "Sensor S solves monitoring.",
            "Closing context.",
        ],
        shared_stream=True,
    )
    response = extracted(doc, "c2")
    answers = [
        response,
        reviewed(),
        response,
        reviewed("unsupported"),
        response,
        reviewed(),
    ]
    result = asyncio.run(
        process_document(
            doc, ReplayProvider(answers), settings=settings(max_model_calls=6)
        )
    )
    assert len(result.assertions) == 1
    assert result.assertions[0].status == "needs_review"
    assert result.assertions[0].verification_status == "unverified"
    assert "conflicting_reviews" in codes(result)
    assert GraphStore._solution_links(doc, result) == []


def test_empty_document_makes_no_model_calls_and_is_skipped_not_failed():
    doc = document()
    doc.chunks = []
    provider = ReplayProvider([])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert list(provider.calls) == []
    # A-10: "failed" claimed an extraction error where nothing was read.
    assert result.run.status == "skipped_no_text"
    assert result.assertions == []
    assert result.run.metadata["coverage"]["total_chunks"] == 0


def test_failed_packet_does_not_discard_a_later_successful_packet():
    doc = document(
        [
            "Unreadable model input in this packet.",
            "Sensor S solves monitoring.",
        ]
    )
    provider = ScriptProvider(
        [
            LLMError("refusal", "Cannot process first packet"),
            extracted(doc, "c2"),
            reviewed(),
        ]
    )
    result = asyncio.run(
        process_document(doc, provider, settings=settings(max_model_calls=3))
    )
    assert result.run.status == "partial"
    assert result.run.metadata["model_calls"] == 3
    assert result.run.metadata["coverage"]["processed_focus_chunk_ids"] == [
        "c2"
    ]
    assert result.run.metadata["coverage"]["unprocessed_chunk_ids"] == ["c1"]
    assert len(result.run.metadata["coverage"]["failed_packet_ids"]) == 1
    assert result.assertions[0].status == "accepted"
    assert result.assertions[0].evidence[0].chunk_id == "c2"


def test_entity_refs_in_nested_qualifiers_and_values_remap_to_the_same_concept():  # noqa: E501
    doc = document(["Sensor S solves monitoring in 1 test."])
    response = extracted(doc)
    response["claims"][0]["qualifiers"] = {
        "groups": [{"nested": {"entity_ref": "sensor"}}],
        "setting": "bench",
    }
    response["claims"][0]["values"] = [
        {"entity_ref": "sensor", "value": 1, "raw": "1 test"}
    ]
    result = asyncio.run(
        process_document(
            doc, ReplayProvider([response, reviewed()]), settings=settings()
        )
    )
    claim = result.assertions[0]
    assert claim.status == "accepted"
    assert (
        claim.qualifiers["groups"][0]["nested"]["entity_ref"]
        == claim.roles["subject"]
    )
    assert claim.values[0]["entity_ref"] == claim.roles["subject"]
    assert claim.values[0]["raw"] == "1 test"


def test_nonfinite_numeric_value_is_gated_before_reviewer():
    doc = document(["Sensor S solves monitoring in 1 test."])
    response = extracted(doc)
    response["claims"][0]["values"] = [
        {"value": float("nan"), "raw": "1 test"}
    ]
    provider = ScriptProvider([response, {"items": []}])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert len(provider.calls) == 2
    assert result.assertions == []
    assert any(
        "nonfinite_numeric_value" in item.get("reasons", [])
        for item in result.run.metadata["issues"]
    )


def test_typed_replay_response_cannot_sanitize_nonfinite_value_into_acceptance():  # noqa: E501
    doc = document(["Sensor S solves monitoring in 1 test."])
    response = extracted(doc)
    response["claims"][0]["values"] = [
        {"value": float("nan"), "raw": "1 test"}
    ]
    provider = ReplayProvider(
        [Extraction.model_validate(response), {"items": []}]
    )
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert not any(claim.status == "accepted" for claim in result.assertions)
    assert len(provider.calls) == 1


def test_reused_provider_run_audit_contains_only_current_attempts():
    doc = document()
    provider = ReplayProvider(
        [extracted(doc), reviewed(), extracted(doc), reviewed()]
    )
    first = asyncio.run(process_document(doc, provider, settings=settings()))
    second = asyncio.run(process_document(doc, provider, settings=settings()))
    assert first.run.run_id != second.run.run_id
    assert len(provider.calls) == 4
    assert len(first.run.metadata["provider_calls"]) == 2
    assert len(second.run.metadata["provider_calls"]) == 2
    assert [
        item["answer_index"] for item in first.run.metadata["provider_calls"]
    ] == [0, 1]
    assert [
        item["answer_index"] for item in second.run.metadata["provider_calls"]
    ] == [2, 3]


def test_failed_context_reextraction_keeps_initial_source_response_in_audit():
    doc = document(
        [
            "Sensor S solves monitoring.",
            "Background only.",
            "Required definition.",
        ]
    )
    first = extracted(doc)
    first["context_requests"] = [
        {
            "tool": "read_chunk",
            "argument": "c3",
            "reason": "Read the required definition.",
        }
    ]
    provider = ScriptProvider(
        [
            first,
            LLMError("timeout", "Offline context extraction failure"),
            reviewed(),
        ]
    )
    result = asyncio.run(
        process_document(doc, provider, settings=settings(max_model_calls=3))
    )
    # B-7: the failed re-extraction used to discard the valid first answer
    # (run "failed", no assertions). The first answer is kept; its context
    # stays unresolved, so the claim waits for review instead of acceptance.
    assert result.run.status == "partial"
    assert [item.status for item in result.assertions] == ["needs_review"]
    assert result.assertions[0].evidence[0].quote == doc.chunks[0].text
    assert {"context_reextraction_failed", "unresolved_context"} <= codes(
        result
    )
    assert [call["stage"] for call in provider.calls] == [
        "extract",
        "extract",
        "review",
    ]
    responses = [
        event
        for event in result.run.trace
        if event["stage"] == "extraction_response"
    ]
    assert len(responses) == 1
    assert responses[0]["packet_id"]
    assert responses[0]["response"] == Extraction.model_validate(
        first
    ).model_dump(mode="python")
    assert (
        responses[0]["response"]["claims"][0]["evidence"][0]["quote"]
        == doc.chunks[0].text
    )
    assert any(
        event["stage"] == "extract"
        and event.get("status") == "failed"
        and event.get("code") == "timeout"
        for event in result.run.trace
    )


def test_technology_named_three_times_reaches_assertions_with_its_claims():
    abstract = (
        "We study a sulfide solid-state electrolyte for batteries. "
        "The sulfide solid-state electrolyte enables fast charging. "
        "A prototype cell with the sulfide solid-state electrolyte was built."
    )
    doc = document([abstract])
    technology = "sulfide solid-state electrolyte"
    assert abstract.count(technology) == 3
    response = {
        "entities": [
            {
                "local_id": "e1",
                "label": technology,
                "kind": "Method",
                "evidence": [{"chunk_id": "c1", "quote": technology}],
            },
            {
                "local_id": "e2",
                "label": "fast charging",
                "kind": "Task",
                "evidence": [{"chunk_id": "c1", "quote": "fast charging"}],
            },
        ],
        "claims": [
            {
                "claim_id": "k1",
                "predicate": "solves_task",
                "roles": {"subject": "e1", "task": "e2"},
                "qualifiers": {},
                "values": [],
                "polarity": "affirmed",
                "modality": "reported",
                "attribution_kind": "author_reported",
                "evidence": [
                    {
                        "chunk_id": "c1",
                        "quote": "The sulfide solid-state electrolyte "
                        "enables fast charging.",
                    }
                ],
            },
            {
                "claim_id": "k2",
                "predicate": "reports_maturity_stage",
                "roles": {"subject": "e1"},
                "qualifiers": {"stage": "prototype"},
                "values": [],
                "polarity": "affirmed",
                "modality": "observed",
                "attribution_kind": "author_reported",
                # Wrong offsets from the model fall back to literal search.
                "evidence": [
                    {
                        "chunk_id": "c1",
                        "quote": "A prototype cell with the sulfide "
                        "solid-state electrolyte was built.",
                        "start": 0,
                        "end": 10,
                    }
                ],
            },
        ],
        "context_requests": [],
    }
    review = {
        "items": [
            {"claim_id": claim, "decision": "supported", "reason": "Checked."}
            for claim in ("k1", "k2")
        ]
    }
    result = asyncio.run(
        process_document(
            doc, RecordingReplay([response, review]), settings=settings()
        )
    )
    concept = next(c for c in result.concepts if c.kind == ConceptKind.METHOD)
    assert concept.preferred_label == technology
    assert len(result.assertions) == 2
    assert all(
        a.status == "accepted" and a.roles["subject"] == concept.concept_id
        for a in result.assertions
    )
    assert result.run.metadata["invalid_entities"] == []
    notes = result.run.metadata["anchoring_notes"]
    assert {
        (note["item"], note["code"], note["occurrences"]) for note in notes
    } == {
        ("entity:e1", "ambiguous_quote_first_occurrence", 3),
        ("claim:k2", "quote_offsets_corrected", 1),
    }
    assert result.run.status == "succeeded"


class FullPacketProvider:
    """Extract two claims per focus chunk; support every reviewed claim."""

    demo = False
    models = {"extract": "offline-extractor", "review": "offline-reviewer"}

    def __init__(self):
        self.calls = []
        self.review_sizes = []

    def generate(self, schema, system, payload, *, stage="extract"):
        self.calls.append({"stage": stage})
        if stage == "review":
            self.review_sizes.append(
                len(json.dumps(payload, ensure_ascii=False))
            )
            return schema.model_validate(
                {
                    "items": [
                        {
                            "claim_id": claim["claim_id"],
                            "decision": "supported",
                            "reason": "Checked.",
                        }
                        for claim in payload["extraction"]["claims"]
                    ]
                }
            )
        texts = {c["chunk_id"]: c["text"] for c in payload["chunks"]}
        entities, claims = [], []
        for chunk_id in payload["packet"]["focus_chunk_ids"]:
            number = chunk_id[1:].zfill(2)
            technology, task = f"array #{number}", f"task #{number}"
            entities += [
                {
                    "local_id": f"t{number}",
                    "label": technology,
                    "kind": "Method",
                    "evidence": [{"chunk_id": chunk_id, "quote": technology}],
                },
                {
                    "local_id": f"g{number}",
                    "label": task,
                    "kind": "Task",
                    "evidence": [{"chunk_id": chunk_id, "quote": task}],
                },
            ]
            sentences = texts[chunk_id].split(" | ")
            claims += [
                {
                    "claim_id": f"k{number}-{index}",
                    "predicate": "solves_task",
                    "roles": {"subject": f"t{number}", "task": f"g{number}"},
                    "qualifiers": {},
                    "values": [],
                    "polarity": "affirmed",
                    "modality": "reported",
                    "attribution_kind": "author_reported",
                    "evidence": [
                        {"chunk_id": chunk_id, "quote": sentences[index]}
                    ],
                }
                for index in range(2)
            ]
        return schema.model_validate(
            {"entities": entities, "claims": claims, "context_requests": []}
        )


def full_text(chunks=30, sentences=20):
    return document(
        [
            " | ".join(
                f"Result {k} of part {n:02d}: the array #{n:02d} solves "
                f"task #{n:02d} in trial {k}."
                for k in range(sentences)
            )
            for n in range(1, chunks + 1)
        ],
        shared_stream=True,
    )


def test_full_size_packets_are_reviewed_with_the_shipped_pipeline_budget():
    doc = full_text()
    shipped = PipelineSettings.from_catalog()
    provider = FullPacketProvider()
    result = asyncio.run(process_document(doc, provider, settings=shipped))
    extract_calls = [c for c in provider.calls if c["stage"] == "extract"]
    assert len(extract_calls) < len(doc.chunks)
    assert provider.review_sizes
    assert max(provider.review_sizes) <= shipped.max_payload_chars
    assert len(result.assertions) == 2 * len(doc.chunks)
    assert {a.status for a in result.assertions} == {"accepted"}
    assert not {"review_contract", "review_payload_budget"} & codes(result)
    assert result.run.status == "succeeded"


def test_claim_too_large_for_review_is_reported_as_budget_not_contract():
    doc = document()
    tight = settings()
    packet = plan_packets(doc, tight).packets[0]
    limit = len(
        json.dumps(build_payload(doc, packet, tight), ensure_ascii=False)
    )
    provider = RecordingReplay([extracted(doc), {"items": []}, {"items": []}])
    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(max_payload_chars=limit)
        )
    )
    assert "review_payload_budget" in codes(result)
    assert "review_contract" not in codes(result)
    assert [item["stage"] for item in provider.payloads] == ["extract"]
    assert result.assertions[0].status == "needs_review"


class ContextRoundProvider(FullPacketProvider):
    """Ask for context once per packet before answering, as models do."""

    def generate(self, schema, system, payload, *, stage="extract"):
        if stage == "extract" and not payload["feedback"]:
            self.calls.append({"stage": stage})
            first = payload["packet"]["focus_chunk_ids"][0]
            return schema.model_validate(
                {
                    "entities": [],
                    "claims": [],
                    "context_requests": [
                        {
                            "tool": "search_chunks",
                            "argument": f"array #{first[1:].zfill(2)}",
                            "reason": "Need the definition.",
                        }
                    ],
                }
            )
        return super().generate(schema, system, payload, stage=stage)


def test_shipped_budget_covers_a_context_round_and_split_review():
    doc = full_text()
    provider = ContextRoundProvider()
    result = asyncio.run(
        process_document(
            doc, provider, settings=PipelineSettings.from_catalog()
        )
    )
    assert provider.review_sizes
    assert "call_budget" not in codes(result)
    assert len(result.assertions) == 2 * len(doc.chunks)
    assert {a.status for a in result.assertions} == {"accepted"}
    assert result.run.status == "succeeded"


def test_run_records_the_pdf_it_read():
    # A-2: "already processed" compares the PDF, not only the version.
    doc = document()
    doc.coverage = "full_text"
    doc.metadata["fulltext"] = {"status": "parsed", "sha256": "pdf-sha"}
    provider = RecordingReplay([extracted(doc), reviewed()])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert result.run.metadata["input_coverage"] == "full_text"
    assert result.run.metadata["input_fulltext_sha256"] == "pdf-sha"


def test_store_reports_what_each_processed_run_read():
    import json as json_module

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def run(self, query, **parameters):
            assert "metadata_json" in query
            return [
                {
                    "id": "version",
                    "metadata_json": json_module.dumps(
                        {
                            "input_coverage": "full_text",
                            "input_fulltext_sha256": "pdf-sha",
                        }
                    ),
                }
            ]

    store = GraphStore.__new__(GraphStore)
    store._driver = type(
        "Driver", (), {"session": lambda self, **_: Session()}
    )()
    store._database = "neo4j"
    found = asyncio.run(store.processed_inputs(["version"]))
    assert found == {
        "version": [{"coverage": "full_text", "fulltext_sha256": "pdf-sha"}]
    }


def test_unanswered_context_request_leaves_only_its_claims_unclear():
    # B-4: one open request used to downgrade every claim of the packet.
    doc = document(["Sensor S solves monitoring. Sensor S solves sorting."])
    response = extracted(doc, claim_id="a")
    response["claims"][0]["evidence"][0]["quote"] = (
        "Sensor S solves monitoring."
    )
    response["entities"].append(
        {
            "local_id": "sorting",
            "label": "sorting",
            "kind": "Task",
            "evidence": [{"chunk_id": "c1", "quote": "sorting"}],
        }
    )
    sorting = deepcopy(response["claims"][0])
    sorting.update(
        claim_id="b",
        roles={"subject": "sensor", "task": "sorting"},
        evidence=[{"chunk_id": "c1", "quote": "Sensor S solves sorting."}],
    )
    response["claims"].append(sorting)
    response["context_requests"] = [
        {
            "tool": "read_chunk",
            "argument": "c9",
            "reason": "Which sorting?",
            "claim_ids": ["b"],
        }
    ]
    review = {
        "items": reviewed(claim_id="a")["items"]
        + reviewed(claim_id="b")["items"]
    }
    provider = RecordingReplay([response, review])
    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(max_context_rounds=0)
        )
    )
    assert "unresolved_context" in codes(result)
    by_quote = {
        item.evidence[0].quote: item.status for item in result.assertions
    }
    assert by_quote["Sensor S solves monitoring."] == "accepted"
    assert by_quote["Sensor S solves sorting."] != "accepted"


def test_context_rounds_of_an_early_packet_leave_calls_for_later_ones():
    # B-4: 3 packets, budget of exactly extraction + review for each.
    doc = document(
        [
            "Sensor S solves monitoring.",
            "Sensor S solves monitoring.",
            "Sensor S solves monitoring.",
        ]
    )
    first = extracted(doc, "c1")
    first["context_requests"] = [
        {"tool": "read_chunk", "argument": "c2", "reason": "More detail."}
    ]
    provider = RecordingReplay(
        [
            first,
            reviewed(),
            extracted(doc, "c2"),
            reviewed(),
            extracted(doc, "c3"),
            reviewed(),
        ]
    )
    result = asyncio.run(
        process_document(doc, provider, settings=settings(max_model_calls=6))
    )
    assert [item["stage"] for item in provider.payloads] == [
        "extract",
        "review",
    ] * 3
    assert result.run.metadata["coverage"]["unprocessed_chunk_ids"] == []
    assert "call_budget" not in codes(result)


def test_one_malformed_element_does_not_cost_the_packet():
    # B-5: extra="forbid" on the whole response dropped every claim.
    doc = document()
    response = extracted(doc)
    response["entities"][0]["confidence"] = 0.9  # an unknown field
    response["entities"].append(
        {
            "local_id": "data",
            "label": "Sensor",
            "kind": "Dataset",  # not a kind of this schema
            "evidence": [{"chunk_id": "c1", "quote": "Sensor"}],
        }
    )
    response["claims"][0]["status"] = "accepted"  # never read from a model
    provider = RecordingReplay([response, reviewed()])
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert [item.status for item in result.assertions] == ["accepted"]
    issue = next(
        item
        for item in result.run.metadata["issues"]
        if item.get("code") == "invalid_items"
    )
    assert issue["items"] == [
        {"field": "entities", "index": 2, "reasons": ["enum"]}
    ]
    assert "invalid_schema" not in codes(result)


def test_json_after_a_preamble_or_inside_a_fence_is_read():
    from lctrend.llm.client import _strip_fence, _validate

    body = json.dumps({"entities": [], "claims": [], "context_requests": []})
    for content in (
        "Here is the JSON:\n```json\n" + body + "\n```\nDone.",
        "Ответ: " + body,
        "```\n" + body + "\n```",
    ):
        assert _validate(Extraction, _strip_fence(content)).claims == []


def test_answer_cut_at_the_output_limit_is_retried_in_halves():
    # B-6: finish_reason=length used to lose the whole packet.
    doc = document(
        ["Sensor S solves monitoring.", "Sensor S solves monitoring."],
        shared_stream=True,
    )
    provider = ScriptProvider(
        [
            LLMError("incomplete_response", "cut at max_tokens"),
            extracted(doc, "c1"),
            reviewed(),
            extracted(doc, "c2"),
            reviewed(),
        ]
    )
    result = asyncio.run(
        process_document(
            doc,
            provider,
            settings=settings(primary_chunks=2, max_model_calls=8),
        )
    )
    assert [call["stage"] for call in provider.calls] == [
        "extract",
        "extract",
        "review",
        "extract",
        "review",
    ]
    assert result.run.metadata["coverage"]["unprocessed_chunk_ids"] == []
    split = next(
        item
        for item in result.run.metadata["issues"]
        if item.get("code") == "incomplete_response"
    )
    assert len(split["split_into"]) == 2
    assert result.assertions and all(
        item.status == "accepted" for item in result.assertions
    )
