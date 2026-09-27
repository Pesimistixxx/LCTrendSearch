"""Hybrid extraction: GLiNER spans are hints and an extra factor, never
evidence.
"""

import asyncio

from test_llm_pipeline import (
    RecordingReplay,
    document,
    extracted,
    reviewed,
    settings,
)

from lctrend.core.models import ConceptKind
from lctrend.llm.pipeline import process_document


class SpanNER:
    """GLiNER-shaped double returning fixed character spans per chunk text."""

    def __init__(self, spans):
        self.spans = spans

    def predict_entities(self, text, labels, threshold=0.5):
        found = []
        for surface, label, score in self.spans:
            start = text.find(surface)
            if start >= 0:
                found.append(
                    {
                        "label": label,
                        "start": start,
                        "end": start + len(surface),
                        "score": score,
                    }
                )
        return found


class BrokenNER:
    def predict_entities(self, text, labels, threshold=0.5):
        raise RuntimeError("model crashed")


TEXT = "Sensor S solves monitoring. Lidar X was also mentioned."


def test_hybrid_hints_corroboration_and_ner_only_technology_candidate():
    doc = document([TEXT])
    provider = RecordingReplay(
        [
            {"stage": "extract", "response": extracted(doc)},
            {"stage": "review", "response": reviewed()},
        ]
    )
    ner = SpanNER(
        [
            ("Sensor S", "technology", 0.91),
            ("monitoring", "method", 0.6),
            ("Lidar X", "technology", 0.83),
            ("mentioned", "technology", 0.4),
        ]
    )
    result = asyncio.run(
        process_document(
            doc,
            provider,
            settings=settings(),
            ner=ner,
            ner_name="gliner-fixture",
        )
    )
    hints = provider.payloads[0]["payload"]["ner_hints"]
    assert [(h["text"], h["kind"]) for h in hints] == [
        ("Sensor S", "Technology"),
        ("Lidar X", "Technology"),
        ("monitoring", "Method"),
    ]
    assert "ner_hints" not in provider.payloads[1]["payload"]
    by_text = {m.surface_text: m for m in result.mentions}
    assert (
        by_text["Sensor S"].mention_role == "entity"
        and by_text["Sensor S"].confidence == 0.91
    )
    assert by_text["monitoring"].confidence is None
    assert by_text["Lidar X"].mention_role == "ner_candidate"
    assert by_text["Lidar X"].type_candidates == [ConceptKind.TECHNOLOGY]
    assert "mentioned" not in by_text
    assert any(
        c.preferred_label == "Lidar X" and c.kind == ConceptKind.TECHNOLOGY
        for c in result.concepts
    )
    assert (
        len(result.assertions) == 1
        and result.assertions[0].status == "accepted"
    )
    ner_audit = result.run.metadata["ner"]
    assert (
        ner_audit["status"] == "ok" and ner_audit["model"] == "gliner-fixture"
    )
    assert (
        ner_audit["corroborated"],
        ner_audit["kind_conflicts"],
        ner_audit["llm_only"],
        ner_audit["ner_candidates_added"],
    ) == (1, 1, 0, 1)


def test_ner_failure_keeps_llm_result():
    doc = document([TEXT])
    provider = RecordingReplay(
        [
            {"stage": "extract", "response": extracted(doc)},
            {"stage": "review", "response": reviewed()},
        ]
    )
    result = asyncio.run(
        process_document(
            doc,
            provider,
            settings=settings(),
            ner=BrokenNER(),
            ner_name="broken",
        )
    )
    assert result.run.metadata["ner"]["status"] == "failed"
    assert result.run.metadata["ner"]["error"] == "RuntimeError"
    assert result.run.status == "succeeded"
    assert provider.payloads[0]["payload"]["ner_hints"] == []
    assert {m.mention_role for m in result.mentions} == {"entity"}


def test_llm_only_run_sends_no_hints():
    doc = document([TEXT])
    provider = RecordingReplay(
        [
            {"stage": "extract", "response": extracted(doc)},
            {"stage": "review", "response": reviewed()},
        ]
    )
    result = asyncio.run(process_document(doc, provider, settings=settings()))
    assert "ner_hints" not in provider.payloads[0]["payload"]
    assert result.run.metadata["ner"] == {"status": "disabled", "model": None}
