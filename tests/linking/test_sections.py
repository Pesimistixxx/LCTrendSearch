"""Data and administrative sections never reach the model."""

import asyncio

from lctrend.core.models import (
    Artifact,
    Chunk,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.linking.sections import skipped_chunks, skipped_role
from lctrend.llm.context import PipelineSettings, plan_packets
from lctrend.llm.pipeline import process_document

DATA_CHUNK = (
    "A. Data Source: Workplace Organization and Labor Force "
    "Characteristics We surveyed senior human resources managers in three "
    "waves in 1995-1996."
)


def chunk(chunk_id, text, order, role=None):
    return Chunk(
        chunk_id=chunk_id,
        kind="fulltext",
        text=text,
        order=order,
        section_path=["fulltext"],
        locator={"section_role": role} if role else {},
    )


def document(*chunks):
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
        chunks=list(chunks),
    )


def test_role_from_the_heading_or_from_a_heading_opening_the_chunk():
    assert skipped_role(chunk("c", "Text", 0, "data_description")) == (
        "data_description"
    )
    assert skipped_role(chunk("c", "Text", 0, "administrative"))
    # A method section describes the technology; it is always read.
    assert skipped_role(chunk("c", "Text", 0, "method")) is None
    assert skipped_role(chunk("c", DATA_CHUNK, 0)) == "data_description"
    assert skipped_role(chunk("c", "3.1 Datasets\nWe use ImageNet.", 0))
    assert skipped_role(chunk("c", "Author contributions: A.B. wrote", 0))
    for text in (
        "Data were collected from 40 plants.",
        "Data-driven fault detection with graph neural networks.",
        "Data analysis: we fit a transformer.",
        "Acknowledgments: funded by the NSF grant 123.",
    ):
        assert skipped_role(chunk("c", text, 0)) is None, text


def test_a_document_made_only_of_skipped_chunks_is_read_whole():
    only_data = document(
        chunk("c1", DATA_CHUNK, 0), chunk("c2", "II. DATA", 1)
    )

    assert skipped_chunks(only_data) == {}


def test_plan_omits_skipped_chunks_and_never_uses_them_as_support():
    record = document(
        chunk("c1", "We propose sensor S for monitoring.", 0),
        chunk("c2", DATA_CHUNK, 1),
        chunk("c3", "Sensor S cut failures by 30%.", 2),
    )
    plan = plan_packets(record, PipelineSettings())

    assert plan.omitted_reasons == {"c2": "skipped_section:data_description"}
    for packet in plan.packets:
        assert "c2" not in packet.focus_chunk_ids + packet.support_chunk_ids


class Recording:
    def __init__(self):
        self.payloads = []

    async def generate(self, schema, system, payload, *, stage="extract"):
        self.payloads.append(payload)
        return schema.model_validate(
            {"entities": [], "claims": []}
            if stage == "extract"
            else {"items": []}
        )


def test_skipped_sections_do_not_make_the_run_partial():
    record = document(
        chunk("c1", "We propose sensor S for monitoring.", 0),
        chunk("c2", DATA_CHUNK, 1),
    )
    model = Recording()
    result = asyncio.run(process_document(record, model))

    assert result.run.status == "succeeded"
    coverage = result.run.metadata["coverage"]
    assert coverage["skipped_section_chunk_ids"] == ["c2"]
    assert coverage["unprocessed_chunk_ids"] == []
    sent = [
        item["text"]
        for payload in model.payloads
        for item in payload["chunks"]
    ]
    assert not any("surveyed senior" in text for text in sent)
