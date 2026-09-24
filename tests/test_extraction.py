from lctrend.models import (
    Artifact,
    Assertion,
    Chunk,
    Concept,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    EvidenceSpan,
    ExtractionResult,
    Mention,
    ProcessingRun,
    ResolutionDecision,
    SourceRef,
    validate_extraction,
)
from lctrend.ner import extract_mentions


def document():
    text = "Сенсор S снижает энергопотребление."
    return DocumentEnvelope(
        document_id="d1",
        document_version_id="v1",
        document_type=DocumentType.ARTICLE,
        title="Test",
        source=SourceRef(source_id="s1", name="fixture", source_type="test", record_id="1"),
        artifact=Artifact(uri="memory://1", sha256="0" * 64, media_type="text/plain"),
        chunks=[Chunk(chunk_id="c1", kind="abstract", text=text, order=0)],
    )


def test_extraction_requires_literal_anchors():
    doc = document()
    concept = Concept(concept_id="tech1", kind=ConceptKind.TECHNOLOGY, preferred_label="Сенсор S")
    result = ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(run_id="r1", parser="fixture", config_hash="x", started_at="2026-01-01T00:00:00Z"),
        mentions=[
            Mention(
                mention_id="m1",
                chunk_id="c1",
                surface_text="Сенсор S",
                start=0,
                end=8,
                type_candidates=[ConceptKind.TECHNOLOGY],
            )
        ],
        concepts=[concept],
        resolutions=[
            ResolutionDecision(
                resolution_id="res1", mention_id="m1", status="accepted", concept_id="tech1"
            )
        ],
        assertions=[
            Assertion(
                assertion_id="a1",
                predicate="qualitative_advantage",
                roles={"subject": "tech1"},
                evidence=[
                    EvidenceSpan(
                        chunk_id="c1",
                        quote=doc.chunks[0].text,
                        start=0,
                        end=len(doc.chunks[0].text),
                    )
                ],
            )
        ],
    )
    validate_extraction(doc, result)


def test_gliner_adapter_anchors_model_output():
    class FakeModel:
        def predict_entities(self, text, labels, threshold):
            return [{"text": "Сенсор S", "start": 0, "end": 8, "label": "technology", "score": 0.9}]

    mentions = extract_mentions(document(), FakeModel())
    assert mentions[0].surface_text == "Сенсор S"
    assert mentions[0].type_candidates == [ConceptKind.TECHNOLOGY]
