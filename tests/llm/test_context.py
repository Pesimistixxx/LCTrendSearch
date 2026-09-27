import json

import pytest

from lctrend.core.models import (
    Artifact,
    Chunk,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.llm.context import (
    ContextBudgetError,
    ContextContractError,
    ContextPacket,
    PipelineSettings,
    build_payload,
    expand_packet,
    plan_packets,
    review_payload,
)


def document(texts, *, sections=None, metadata=None, locators=None):
    return DocumentEnvelope(
        document_id="d",
        document_version_id="v",
        document_type=DocumentType.REPORT,
        title="Original title",
        source=SourceRef(
            source_id="s", name="Source", source_type="local", record_id="r"
        ),
        artifact=Artifact(
            uri="local://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        metadata=metadata or {},
        chunks=[
            Chunk(
                chunk_id=f"c{i}",
                kind="paragraph",
                text=text,
                order=i,
                section_path=(sections[i] if sections else ["Section"]),
                locator=(locators[i] if locators else {"line_start": i + 1}),
            )
            for i, text in enumerate(texts)
        ],
    )


def packet(*ids):
    return ContextPacket(packet_id="packet", focus_chunk_ids=list(ids))


def test_settings_come_from_catalog_and_validate_interdependent_limits():
    settings = PipelineSettings.from_catalog()
    assert settings.primary_chunks == 10
    assert settings.max_model_calls == 8
    with pytest.raises(ValueError):
        PipelineSettings(primary_chunks=3, max_context_chunks=2)


def test_call_budget_scales_with_planned_packets_up_to_the_document_cap():
    fixed = PipelineSettings(max_model_calls=8)
    assert fixed.call_limit(50) == 8
    scaled = PipelineSettings(
        max_model_calls=8,
        model_calls_per_packet=3,
        max_document_model_calls=60,
    )
    assert scaled.call_limit(1) == 8
    assert scaled.call_limit(10) == 30
    assert scaled.call_limit(50) == 60


def test_payload_keeps_original_chunks_and_only_allowed_metadata():
    doc = document(
        ["Authors cannot claim a benefit."],
        metadata={
            "summary": "Source summary",
            "api_key": "do-not-send",
            "arbitrary_blob": "x" * 100000,
        },
    )
    payload = build_payload(doc, packet("c0"), PipelineSettings())
    assert payload["chunks"] == [doc.chunks[0].model_dump(mode="json")]
    assert payload["document"]["metadata"] == {"summary": "Source summary"}
    assert "do-not-send" not in json.dumps(payload)
    assert "targets_problem" in payload["predicate_contract"]


def test_domain_and_external_identifiers_are_navigation_metadata_not_source_text():  # noqa: E501
    from lctrend.core.models import Domain, ExternalId

    doc = document(["Source body."])
    doc.domains = [Domain(domain_id="domain", name="Computer vision")]
    doc.identifiers = [ExternalId(scheme="doi", value="10.1/example")]
    payload = build_payload(doc, packet("c0"), PipelineSettings())
    assert payload["document"]["domains"] == ["Computer vision"]
    assert payload["document"]["identifiers"] == ["doi:10.1/example"]
    assert payload["chunks"][0]["text"] == "Source body."


def test_primary_coverage_is_partitioned_by_section_and_support_is_separate():
    doc = document(
        ["A", "B", "C", "D"], sections=[["One"], ["One"], ["Two"], ["Two"]]
    )
    settings = PipelineSettings(primary_chunks=2)
    plan = plan_packets(doc, settings)
    assert [item.focus_chunk_ids for item in plan.packets] == [
        ["c0", "c1"],
        ["c2", "c3"],
    ]
    assert plan.packets[0].support_chunk_ids == ["c2"]
    assert plan.packets[1].support_chunk_ids == ["c1"]
    assert plan.omitted_chunk_ids == []
    assert doc.chunks[2].text == "C"


def test_overlarge_raw_chunk_is_explicitly_omitted_without_being_sliced():
    doc = document(["small", "larger-than-budget"])
    settings = PipelineSettings(primary_chunks=1, max_source_chars=5)
    plan = plan_packets(doc, settings)
    assert plan.omitted_chunk_ids == ["c1"]
    assert plan.omitted_reasons["c1"] == "source_chars_limit"
    assert plan.support_omissions[plan.packets[0].packet_id] == ["c1"]
    assert doc.chunks[1].text == "larger-than-budget"
    assert (
        build_payload(doc, plan.packets[0], settings)["chunks"][0]["text"]
        == "small"
    )


def test_full_payload_budget_includes_metadata_and_predicate_contract():
    doc = document(["tiny"], metadata={"summary": "m" * 10000})
    settings = PipelineSettings(max_payload_chars=5000)
    plan = plan_packets(doc, settings)
    assert plan.omitted_chunk_ids == ["c0"]
    assert plan.omitted_reasons["c0"] == "payload_chars_limit"
    with pytest.raises(ContextBudgetError):
        build_payload(doc, packet("c0"), settings)


def test_navigation_index_is_bounded_and_its_reduction_is_visible():
    doc = document(["Original"] * 70)
    payload = build_payload(
        doc, packet("c50"), PipelineSettings(max_map_entries=3)
    )
    assert len(payload["document_map"]["entries"]) == 3
    assert payload["document_map"]["omitted_entries"] == 67
    assert payload["document_map"]["limit_reason"] == "max_map_entries"
    assert "c50" in {
        entry["chunk_id"] for entry in payload["document_map"]["entries"]
    }
    baseline = build_payload(
        doc, packet("c50"), PipelineSettings(max_map_entries=0)
    )
    size = len(json.dumps(baseline, ensure_ascii=False))
    tighter = build_payload(
        doc, packet("c50"), PipelineSettings(max_payload_chars=size + 80)
    )
    assert len(json.dumps(tighter, ensure_ascii=False)) <= size + 80
    assert tighter["document_map"]["limit_reason"] == "payload_budget"
    assert tighter["chunks"][0]["text"] == "Original"


def test_context_tools_are_literal_document_local_and_bounded():
    doc = document(
        ["Focus", "Definition [A]", "More definition [A]", "Another [A]"]
    )
    requests = [
        {"tool": "search_chunks", "argument": "[A]", "reason": "definition"},
        {"tool": "read_chunk", "argument": "foreign-id", "reason": "invalid"},
        {"tool": "shell", "argument": "something", "reason": "not allowed"},
        {"tool": "read_chunk", "argument": "c3", "reason": "over limit"},
    ]
    result, outcomes = expand_packet(
        doc,
        packet("c0"),
        requests,
        PipelineSettings(search_limit=1, max_requests_per_round=3),
    )
    assert result.support_chunk_ids == ["c1"]
    assert [item["status"] for item in outcomes] == [
        "added",
        "unknown_chunk",
        "unsupported_tool",
        "request_limit",
    ]
    assert doc.chunks[1].text == "Definition [A]"


def test_expansion_respects_source_and_full_serialized_payload_limits():
    doc = document(
        ["A", "B"], locators=[{}, {"arbitrary_source_locator": "q" * 6000}]
    )
    baseline = build_payload(
        doc, packet("c0"), PipelineSettings(max_map_entries=0)
    )
    budget = len(json.dumps(baseline, ensure_ascii=False)) + 500
    result, outcomes = expand_packet(
        doc,
        packet("c0"),
        [{"tool": "read_chunk", "argument": "c1"}],
        PipelineSettings(max_map_entries=0, max_payload_chars=budget),
    )
    assert result.support_chunk_ids == []
    assert outcomes[0]["status"] == "context_budget"
    assert outcomes[0]["budget_reason"] == "payload_chars_limit"
    result, outcomes = expand_packet(
        document(["AA", "BB"]),
        packet("c0"),
        [{"tool": "read_chunk", "argument": "c1"}],
        PipelineSettings(max_source_chars=3),
    )
    assert outcomes[0]["budget_reason"] == "source_chars_limit"


def test_foreign_or_duplicate_chunks_are_not_packet_evidence():
    doc = document(["A"])
    with pytest.raises(ContextContractError):
        build_payload(doc, packet("missing"), PipelineSettings())
    with pytest.raises(ValueError):
        ContextPacket(
            packet_id="x", focus_chunk_ids=["c0"], support_chunk_ids=["c0"]
        )


def test_reviewer_gets_complete_cited_originals_and_neighbors_when_they_fit():
    doc = document(
        [
            "Negation before.",
            "Authors do not claim success.",
            "Conditions after.",
        ]
    )
    extraction = {
        "claims": [{"evidence": [{"chunk_id": "c1", "quote": "success"}]}]
    }
    payload = review_payload(doc, extraction, ["c1"], PipelineSettings())
    assert payload["chunks"] == [
        item.model_dump(mode="json") for item in doc.chunks
    ]
    assert payload["review_context"]["required_chunk_ids"] == ["c1"]
    assert payload["review_context"]["neighbor_chunk_ids"] == ["c0", "c2"]


def test_reviewer_never_drops_required_cited_chunks_to_fit_budget():
    doc = document(["A" * 100, "B" * 100])
    extraction = {"claims": [{"evidence": [{"chunk_id": "c0", "quote": "A"}]}]}
    payload = review_payload(
        doc, extraction, ["c0"], PipelineSettings(max_source_chars=100)
    )
    assert payload["review_context"]["omitted_neighbor_chunk_ids"] == ["c1"]
    assert payload["chunks"][0]["text"] == "A" * 100
    with pytest.raises(ContextBudgetError):
        review_payload(
            doc, extraction, ["c0"], PipelineSettings(max_source_chars=99)
        )
    with pytest.raises(ContextContractError):
        review_payload(doc, extraction, ["c1"], PipelineSettings())


def test_rejected_chunk_is_not_primary_support_or_reviewer_context():
    doc = document(["Valid source", "Rejected parse result"])
    doc.chunks[1].parse_status = "rejected"
    settings = PipelineSettings(primary_chunks=1)
    plan = plan_packets(doc, settings)
    assert plan.omitted_reasons == {"c1": "parse_rejected"}
    assert plan.packets[0].focus_chunk_ids == ["c0"]
    assert plan.packets[0].support_chunk_ids == []
    payload = review_payload(
        doc,
        {"claims": [{"evidence": [{"chunk_id": "c0", "quote": "Valid"}]}]},
        ["c0"],
        settings,
    )
    assert [chunk["chunk_id"] for chunk in payload["chunks"]] == ["c0"]
    assert payload["review_context"]["omitted_neighbor_chunk_ids"] == ["c1"]
    with pytest.raises(ContextContractError, match="rejected"):
        build_payload(doc, packet("c1"), settings)


def test_requested_rejected_chunk_is_refused_and_local_search_skips_it():
    doc = document(
        [
            "Focus",
            "Definition from invalid OCR",
            "Definition from valid source",
        ]
    )
    doc.chunks[1].parse_status = "rejected"
    result, outcomes = expand_packet(
        doc,
        packet("c0"),
        [
            {"tool": "read_chunk", "argument": "c1"},
            {"tool": "search_chunks", "argument": "Definition"},
        ],
        PipelineSettings(search_limit=1),
    )
    assert outcomes[0]["status"] == "rejected_chunk"
    assert outcomes[0]["omitted_chunk_ids"] == ["c1"]
    assert result.support_chunk_ids == ["c2"]
