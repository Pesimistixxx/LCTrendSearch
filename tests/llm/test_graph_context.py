"""Graph context requests and original evidence boundaries, offline."""

import asyncio
import json

import pytest
from test_llm_pipeline import (
    RecordingReplay,
    document,
    extracted,
    reviewed,
    settings,
)

from lctrend.llm.context import (
    ContextPacket,
    expand_context,
    review_payload,
)
from lctrend.llm.contracts import ContextRequest, Extraction
from lctrend.llm.pipeline import process_document
from lctrend.llm.validation import validate_local_extraction

pytestmark = pytest.mark.legacy_technology_entities


def related(text="Sensor S background from an earlier report."):
    return {
        "chunk_id": "foreign",
        "text": text,
        "document_id": "earlier",
        "document_version_id": "earlier-v1",
        "title": "Earlier report",
        "kind": "paragraph",
        "locator": {"page": 3},
    }


def request():
    return ContextRequest(
        tool="search_graph",
        argument="Sensor S",
        reason="Resolve technology context",
    )


def test_graph_request_rereads_source_and_reviewer_sees_same_provenance():
    doc = document()
    first = extracted(doc)
    first["context_requests"] = [request().model_dump()]
    provider = RecordingReplay([first, extracted(doc), reviewed()])
    calls = []

    async def reader(**kwargs):
        calls.append(kwargs)
        return [related()]

    result = asyncio.run(
        process_document(
            doc, provider, settings=settings(), context_reader=reader
        )
    )
    assert calls[0]["exclude_version_id"] == doc.document_version_id
    assert calls[0]["query"] == "Sensor S"
    assert result.run.status == "succeeded"
    assert result.assertions[0].status == "accepted"
    for call in provider.payloads[1:]:
        payload = call["payload"]
        assert payload["related_context"]["chunks"] == [related()]
        assert [item["chunk_id"] for item in payload["chunks"]] == ["c1"]
    assert any(
        item.get("outcomes", [{}])[0].get("added_chunk_ids") == ["foreign"]
        for item in result.run.trace
        if item.get("stage") == "context"
    )


def test_foreign_chunk_cannot_support_current_document_assertion():
    doc = document()
    first, replacement = extracted(doc), extracted(doc)
    first["context_requests"] = [request().model_dump()]
    replacement["claims"][0]["evidence"] = [
        {"chunk_id": "foreign", "quote": related()["text"]}
    ]
    provider = RecordingReplay([first, replacement])
    result = asyncio.run(
        process_document(
            doc,
            provider,
            settings=settings(),
            context_reader=lambda **kwargs: [related()],
        )
    )
    assert not result.assertions
    assert result.run.metadata["unresolved_claims"]
    assert [item["stage"] for item in provider.payloads] == [
        "extract",
        "extract",
    ]


def test_empty_graph_search_can_complete_after_explicit_reread():
    doc = document()
    first = extracted(doc)
    first["context_requests"] = [request().model_dump()]
    result = asyncio.run(
        process_document(
            doc,
            RecordingReplay([first, extracted(doc), reviewed()]),
            settings=settings(),
            context_reader=lambda **kwargs: [],
        )
    )
    assert result.run.status == "succeeded"
    assert (
        next(item for item in result.run.trace if item["stage"] == "context")[
            "outcomes"
        ][0]["status"]
        == "no_match"
    )


def test_graph_limits_keep_full_chunks_and_record_omissions():
    doc = document()
    packet = ContextPacket(packet_id="p", focus_chunk_ids=["c1"])
    _, chunks, outcomes = asyncio.run(
        expand_context(
            doc,
            packet,
            [request()],
            settings(),
            [],
            lambda **kwargs: [related("x" * 7000)],
        )
    )
    assert chunks == []
    assert outcomes[0]["status"] == "context_budget"
    assert outcomes[0]["omitted_chunk_ids"] == ["foreign"]


def test_graph_reader_error_and_missing_reader_have_visible_outcomes():
    doc = document()
    packet = ContextPacket(packet_id="p", focus_chunk_ids=["c1"])

    def failed(**kwargs):
        raise RuntimeError("private connection details")

    for reader, expected in [
        (None, "graph_unavailable"),
        (failed, "graph_read_failed"),
    ]:
        _, _, outcomes = asyncio.run(
            expand_context(doc, packet, [request()], settings(), [], reader)
        )
        assert outcomes[0]["status"] == expected
        assert "private" not in str(outcomes)


def test_reviewer_retains_requested_nonadjacent_support():
    doc = document(
        ["Sensor S solves monitoring.", "Other file.", "Important definition."]
    )
    payload = review_payload(doc, extracted(doc), ["c1", "c3"], settings())
    assert {item["chunk_id"] for item in payload["chunks"]} == {"c1", "c3"}


def test_reader_cannot_return_primary_version_as_foreign_context():
    doc = document()
    row = related()
    row["document_version_id"] = doc.document_version_id
    packet = ContextPacket(packet_id="p", focus_chunk_ids=["c1"])
    _, chunks, outcomes = asyncio.run(
        expand_context(
            doc, packet, [request()], settings(), [], lambda **kwargs: [row]
        )
    )
    assert not chunks
    assert outcomes[0]["status"] == "invalid_graph_context"


def test_invalid_graph_response_is_an_optional_read_failure():
    doc = document()
    packet = ContextPacket(packet_id="p", focus_chunk_ids=["c1"])
    _, chunks, outcomes = asyncio.run(
        expand_context(
            doc, packet, [request()], settings(), [], lambda **kwargs: None
        )
    )
    assert not chunks
    assert outcomes[0]["status"] == "invalid_graph_context"


def test_optional_graph_rows_cannot_prevent_required_source_review():
    doc = document(["Sensor S solves monitoring. " + "x" * 800])
    extraction = extracted(doc)
    extraction["claims"][0]["values"] = [{"raw": doc.chunks[0].text}] * 20
    anchored, issues = validate_local_extraction(
        doc, Extraction.model_validate(extraction), ["c1"]
    )
    assert not issues
    source_review = review_payload(
        doc, anchored.model_dump(mode="python"), ["c1"], settings()
    )
    limited = settings(
        max_payload_chars=len(json.dumps(source_review, ensure_ascii=False))
        + 500
    )
    first = extracted(doc)
    first["context_requests"] = [request().model_dump()]
    provider = RecordingReplay([first, extraction, reviewed()])
    result = asyncio.run(
        process_document(
            doc,
            provider,
            settings=limited,
            context_reader=lambda **kwargs: [related("x" * 3000)],
        )
    )
    assert result.run.status == "succeeded"
    assert provider.payloads[-1]["stage"] == "review"
    reviewer = provider.payloads[-1]["payload"]
    assert reviewer["review_context"]["omitted_related_chunk_ids"] == [
        "foreign"
    ]
    assert reviewer["chunks"][0]["text"] == doc.chunks[0].text
    assert any(
        item.get("context", {}).get("omitted_related_chunk_ids") == ["foreign"]
        for item in result.run.trace
        if item["stage"] == "review_context"
    )
