"""Registry concepts given to the extractor as reference names."""

import asyncio

import numpy as np

from lctrend.core.models import (
    Artifact,
    Chunk,
    Concept,
    ConceptKind,
    ConceptName,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
    stable_id,
)
from lctrend.extraction.resolver import concept_text, normalize_name
from lctrend.linking.known import KnownConcepts
from lctrend.llm.context import PipelineSettings, build_payload, plan_packets
from lctrend.llm.pipeline import process_document

OPTIONS = {
    "enabled": True,
    "limit": 3,
    "kinds": ["Technology", "Method", "Task"],
    "semantic": True,
    "semantic_min_cosine": 0.5,
    "max_chunk_chars": 2000,
    "max_names": 3,
    "max_definition_chars": 40,
}


def concept(label, kind=ConceptKind.TECHNOLOGY, **fields):
    return Concept(
        concept_id=stable_id("concept", label),
        kind=kind,
        preferred_label=label,
        **fields,
    )


GNN = concept(
    "graph neural network",
    definition="neural network over graph-structured data",
    names=[
        ConceptName(
            name_id="n1",
            text="GNN",
            normalized_text="gnn",
            status="accepted",
        )
    ],
)
RAG = concept("retrieval-augmented generation")
BATTERY = concept("solid-state battery")
FRAUD = concept("fraud detection", ConceptKind.TASK)
USA = concept("United States", ConceptKind.COUNTRY)


class Semantic:
    """Unit vectors on fixed axes; a text about retrieval is near RAG."""

    embedding_model_name = "fixture"

    def __init__(self):
        self.axes = {
            normalize_name(concept_text(RAG)): [1.0, 0.0, 0.0],
            normalize_name(concept_text(BATTERY)): [0.0, 1.0, 0.0],
        }
        self.embedded = []

    def cached_vector(self, text):
        return self.axes.get(normalize_name(text))

    def embed(self, texts, cache=True):
        self.embedded.extend(texts)
        return [
            [0.8, 0.1, 0.59] if "retriev" in text else [0.0, 0.0, 1.0]
            for text in texts
        ]


def test_named_concepts_come_first_then_similar_ones():
    known = KnownConcepts([GNN, RAG, BATTERY, FRAUD, USA], Semantic(), OPTIONS)
    found = known.lookup(
        ["GNNs detect fraud detection cases; we retrieve passages."]
    )

    assert [(item["match"], item["label"]) for item in found] == [
        ("name", "graph neural network"),
        ("name", "fraud detection"),
        ("similar", "retrieval-augmented generation"),
    ]
    assert found[0]["names"] == ["GNN"]
    assert found[0]["definition"] == GNN.definition[:40]
    assert found[2]["cosine"] == 0.8


def test_kinds_outside_the_list_and_far_vectors_are_left_out():
    known = KnownConcepts([USA, BATTERY], Semantic(), OPTIONS)

    # A country is not a reference name; the battery vector is far.
    assert known.lookup(["United States market for sensors."]) == []


def test_without_a_semantic_layer_only_names_are_matched():
    known = KnownConcepts([GNN, RAG], None, OPTIONS)

    assert [item["label"] for item in known.lookup(["we retrieve"])] == []
    assert [item["label"] for item in known.lookup(["a GNN"])] == [
        "graph neural network"
    ]


def document(text):
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
        chunks=[Chunk(chunk_id="c1", kind="abstract", text=text, order=0)],
    )


def test_the_block_is_trimmed_before_it_breaks_the_payload_budget():
    record = document("GNNs detect fraud.")
    settings = PipelineSettings()
    packet = plan_packets(record, settings).packets[0]
    size = len(str(build_payload(record, packet, settings)))
    entries = [
        {"label": f"concept {index}", "kind": "Technology", "match": "name"}
        for index in range(200)
    ]
    tight = settings.model_copy(update={"max_payload_chars": size + 400})
    payload = build_payload(record, packet, tight, known_concepts=entries)

    kept = payload["known_concepts"]["concepts"]
    assert 0 < len(kept) < len(entries)
    assert "never evidence" in payload["known_concepts"]["purpose"]


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


def test_the_extractor_receives_known_concepts_of_its_packet(monkeypatch):
    monkeypatch.setattr("lctrend.linking.known.settings", lambda: OPTIONS)
    monkeypatch.setattr("lctrend.linking.known.enabled", lambda: True)
    model = Recording()
    result = asyncio.run(
        process_document(
            document("GNNs detect fraud."), model, registry=[GNN, RAG]
        )
    )

    [payload] = model.payloads
    assert [
        item["label"] for item in payload["known_concepts"]["concepts"]
    ] == ["graph neural network"]
    trace = [
        item for item in result.run.trace if item["stage"] == "known_concepts"
    ]
    assert trace[0]["concepts"] == [
        ["name", "Technology", "graph neural network"]
    ]


def test_the_matrix_uses_only_cached_vectors():
    semantic = Semantic()
    known = KnownConcepts([GNN, RAG, BATTERY], semantic, OPTIONS)

    assert known.matrix.shape == (2, 3)
    assert np.allclose(np.linalg.norm(known.matrix, axis=1), 1.0)
