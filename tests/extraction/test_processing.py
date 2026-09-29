"""Shared LLM extraction entry point, without external services."""

import asyncio

import pytest

from lctrend.core.models import (
    Artifact,
    Chunk,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.extraction.processing import process_material
from lctrend.llm.client import LLMError, ReplayProvider

pytestmark = pytest.mark.legacy_technology_entities


def document():
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="v1",
        document_type=DocumentType.REPORT,
        title="Sensor study",
        source=SourceRef(
            source_id="fixture",
            name="Fixture",
            source_type="test",
            record_id="1",
        ),
        artifact=Artifact(
            uri="memory://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        chunks=[
            Chunk(
                chunk_id="c1",
                kind="paragraph",
                text="Sensor S solves monitoring.",
                order=0,
            )
        ],
    )


def responses():
    return [
        {
            "entities": [
                {
                    "local_id": "sensor",
                    "label": "Sensor S",
                    "kind": "Technology",
                    "evidence": [{"chunk_id": "c1", "quote": "Sensor S"}],
                },
                {
                    "local_id": "task",
                    "label": "monitoring",
                    "kind": "Task",
                    "evidence": [{"chunk_id": "c1", "quote": "monitoring"}],
                },
            ],
            "claims": [
                {
                    "claim_id": "claim",
                    "predicate": "solves_task",
                    "roles": {"subject": "sensor", "task": "task"},
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "quote": "Sensor S solves monitoring.",
                        }
                    ],
                }
            ],
        },
        {
            "items": [
                {
                    "claim_id": "claim",
                    "decision": "supported",
                    "reason": "The quoted text supports it.",
                }
            ]
        },
    ]


def test_llm_mode_processes_document():
    result = asyncio.run(
        process_material(document(), provider=ReplayProvider(responses()))
    )
    assert result.run.parser == "llm_packets"
    assert result.run.status == "succeeded"
    assert result.assertions[0].status == "accepted"
    assert "ner" not in result.run.metadata


def test_none_mode_skips_extraction():
    result = asyncio.run(process_material(document(), mode="none"))
    assert result.run.parser == "metadata"
    assert result.run.metadata["extraction"] == "disabled"


def test_failed_llm_has_no_fallback():
    class BrokenProvider:
        def generate(self, *args, **kwargs):
            raise LLMError("auth", "Credentials rejected", retryable=False)

    result = asyncio.run(
        process_material(document(), provider=BrokenProvider())
    )
    assert result.run.status == "failed"
    assert result.assertions == []
    assert result.mentions == []


@pytest.mark.parametrize("mode", ["unsupported", "search"])
def test_invalid_mode_fails_before_starting_work(mode):
    with pytest.raises(ValueError, match="mode must be llm or none"):
        asyncio.run(process_material(document(), mode=mode))
