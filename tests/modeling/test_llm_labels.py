import csv
import json

from lctrend.modeling.dataset.llm_labels import (
    ProvisionalAssessment,
    packets,
    validate_assessment,
)


def _ref(document_id, when):
    return {"document_id": document_id, "date": when, "title": document_id}


def test_packet_uses_only_dated_references_and_skips_empty_pilot(tmp_path):
    path = tmp_path / "review.csv"
    rows = [
        {
            "technology_id": "t",
            "technology": "Technology",
            "family_id": "f",
            "snapshot_date": "2020-01-01",
            "horizon_12m_end": "2021-01-01",
            "horizon_end": "2023-01-01",
            "calendar_complete_12m": "True",
            "calendar_complete_36m": "True",
            "document_count": "2",
            "recent_documents": json.dumps([_ref("a", "2019-01-01")]),
        },
        {
            "technology_id": "t",
            "technology": "Technology",
            "family_id": "f",
            "snapshot_date": "2021-01-01",
            "horizon_12m_end": "2022-01-01",
            "horizon_end": "2024-01-01",
            "calendar_complete_12m": "True",
            "calendar_complete_36m": "False",
            "document_count": "2",
            "recent_documents": json.dumps([_ref("a", "2019-01-01")]),
        },
        {
            "technology_id": "t",
            "technology": "Technology",
            "family_id": "f",
            "snapshot_date": "2022-01-01",
            "horizon_12m_end": "2023-01-01",
            "horizon_end": "2025-01-01",
            "calendar_complete_12m": "True",
            "calendar_complete_36m": "False",
            "document_count": "3",
            "recent_documents": json.dumps([_ref("b", "2021-06-01")]),
        },
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    pilot = list(packets(path))
    assert len(pilot) == 1
    assert pilot[0][1]["followup_0_12m"] == []
    assert pilot[0][1]["followup_13_36m"][0]["document_id"] == "b"


def test_validation_abstains_without_timed_citations():
    packet = {
        "at_t": [_ref("a", "2020-01-01")],
        "followup_0_12m": [_ref("b", "2020-06-01")],
        "followup_13_36m": [],
        "calendar_complete_36m": False,
    }
    answer = ProvisionalAssessment(
        state_at_t="early",
        signal_12m="1",
        trend_12m="1",
        signal_36m="1",
        trend_36m="1",
        evidence_at_t_document_ids=["a"],
        evidence_future_document_ids=[],
        rationale_at_t="Provisional",
        rationale_future="Uncertain",
    )
    checked = validate_assessment(answer, packet)
    assert checked["signal_12m"] == "unknown"
    assert checked["signal_36m"] == "unknown"
    answer.evidence_future_document_ids = ["b"]
    checked = validate_assessment(answer, packet)
    assert checked["signal_12m"] == "1"
    assert checked["signal_36m"] == "unknown"
