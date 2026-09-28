"""Retries after HTTP 429 do not spend a document's model call budget."""

import asyncio

from lctrend.core.models import (
    Artifact,
    Chunk,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.llm.client import LLMError
from lctrend.llm.context import PipelineSettings
from lctrend.llm.pipeline import process_document


def document():
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="v1",
        document_type=DocumentType.ARTICLE,
        title="Paper",
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
            Chunk(chunk_id="c1", kind="abstract", text="Sensor S.", order=0)
        ],
    )


class Limited:
    """Answers 429 ``limited`` times, then an empty extraction."""

    def __init__(self, limited):
        self.limited = limited
        self.requests = 0

    async def generate(self, schema, system, payload, *, stage="extract"):
        self.requests += 1
        if self.limited:
            self.limited -= 1
            raise LLMError(
                "http_error",
                "LLM endpoint returned HTTP 429",
                True,
                status=429,
            )
        return schema.model_validate({"entities": [], "claims": []})


def test_rate_limited_retries_leave_the_budget_for_real_calls():
    settings = PipelineSettings(
        max_model_calls=2,
        max_retries=3,
        retry_delay_seconds=0,
        max_retry_delay_seconds=0,
    )
    model = Limited(limited=3)
    result = asyncio.run(
        process_document(document(), model, settings=settings)
    )

    assert model.requests == 4
    assert result.run.status == "succeeded"
    assert result.run.metadata["model_calls"] == 1
