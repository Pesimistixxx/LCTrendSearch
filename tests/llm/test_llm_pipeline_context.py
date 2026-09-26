"""Regressions for canonical overlap evidence and the fully framed context
budget.
"""

import json

import pytest
from test_llm_pipeline import (
    RecordingReplay,
    document,
    extracted,
    reviewed,
    settings,
)

from lctrend.llm.client import ReplayProvider
from lctrend.llm.context import build_payload, expand_packet, plan_packets
from lctrend.llm.pipeline import process_document


@pytest.mark.parametrize(
    "shared_source,expected_assertions,expected_mentions",
    [(True, 1, 2), (False, 2, 4)],
)
def test_overlap_dedup_uses_original_coordinates_and_never_merges_distinct_streams(  # noqa: E501
    shared_source, expected_assertions, expected_mentions
):
    doc = document(
        ["Sensor S solves monitoring.", "Sensor S solves monitoring. Tail."],
        shared_stream=True,
    )
    for index, chunk in enumerate(doc.chunks):
        stream = (
            "original-stream"
            if shared_source
            else f"independent-stream-{index}"
        )
        chunk.locator.update(
            source_stream_id=stream,
            source_start=0,
            source_end=len(chunk.text),
            source_segments=[
                {
                    "chunk_start": 0,
                    "chunk_end": len(chunk.text),
                    "source_start": 0,
                    "source_end": len(chunk.text),
                }
            ],
        )
    first = extracted(doc, "c1")
    second = extracted(doc, "c2")
    second["claims"][0]["evidence"][0]["quote"] = first["claims"][0][
        "evidence"
    ][0]["quote"]
    result = process_document(
        doc,
        ReplayProvider([first, reviewed(), second, reviewed()]),
        settings=settings(max_model_calls=4),
    )
    assert len(result.assertions) == expected_assertions
    assert len(result.mentions) == expected_mentions
    assert all(claim.status == "accepted" for claim in result.assertions)
    assert result.run.metadata["coverage"]["processed_focus_chunk_ids"] == [
        "c1",
        "c2",
    ]
    if shared_source:
        # Keep a real validated anchor, even when another chunk contains the
        # same source span.
        assert result.assertions[0].evidence[0].chunk_id == "c1"
        assert all(mention.chunk_id == "c1" for mention in result.mentions)


def test_feedback_overflow_keeps_initial_anchored_candidate_without_extra_extraction_call():  # noqa: E501
    # Enough requested source text for the old candidate's reviewer payload to
    # fit below the expanded extraction budget; only extra feedback overflows.
    doc = document(
        [
            "Sensor S solves monitoring.",
            "Required source definition. " + "x" * 3000,
        ]
    )
    request = {
        "tool": "read_chunk",
        "argument": "c2",
        "reason": "Need source definition.",
    }
    unlimited = settings(max_map_entries=0, max_model_calls=3)
    packet = plan_packets(doc, unlimited).packets[0]
    expanded, outcomes = expand_packet(doc, packet, [request], unlimited)
    assert outcomes[0]["status"] == "added"
    payload = build_payload(doc, expanded, unlimited)
    # Original context fits exactly; framing the replacement with feedback
    # does not.
    exact_budget = len(json.dumps(payload, ensure_ascii=False))
    limited = unlimited.model_copy(update={"max_payload_chars": exact_budget})
    first = extracted(doc)
    first["context_requests"] = [request]
    provider = RecordingReplay([first, reviewed()])
    result = process_document(doc, provider, settings=limited)
    assert [call["stage"] for call in provider.calls] == ["extract", "review"]
    assert result.run.metadata["model_calls"] == 2
    assert len(result.assertions) == 1
    assert result.assertions[0].status == "needs_review"
    assert result.assertions[0].verification_status == "unverified"
    assert result.assertions[0].evidence[0].chunk_id == "c1"
    assert any(
        issue.get("code") == "context_payload_budget"
        for issue in result.run.metadata["issues"]
    )
    assert result.run.metadata["coverage"]["processed_focus_chunk_ids"] == [
        "c1"
    ]
