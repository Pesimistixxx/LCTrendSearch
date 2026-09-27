"""Semantic layer in LLM runs: candidates, fallback and stored vectors."""

import asyncio

from lctrend.core.models import (
    SEMANTIC_CANDIDATE_METHOD,
    Artifact,
    Chunk,
    Concept,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.extraction.resolver import SemanticDeduplicator
from lctrend.graph.store import GraphStore
from lctrend.llm.client import ReplayProvider
from lctrend.llm.context import PipelineSettings
from lctrend.llm.pipeline import process_document
from tests.llm.test_parties_and_maturity import Transaction

TEXT = "Carbon conversion solves monitoring."
VECTORS = {
    "carbon conversion": [1.0, 0.1],
    "carbon capture": [1.0, 0.0],
}


class FakeEmbedder:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def embed(self, texts, model):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embeddings endpoint unavailable")
        return [VECTORS.get(text.casefold(), [0.0, 1.0]) for text in texts]


def semantic(fail=False):
    deduplicator = SemanticDeduplicator(embedder=FakeEmbedder(fail))
    deduplicator._decision_score = lambda left, right: 0.9
    return deduplicator


def document():
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="version",
        document_type=DocumentType.REPORT,
        title="Carbon",
        source=SourceRef(
            source_id="fixture",
            name="fixture",
            source_type="test",
            record_id="1",
        ),
        artifact=Artifact(
            uri="memory://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        chunks=[Chunk(chunk_id="c1", kind="paragraph", text=TEXT, order=0)],
    )


def answers():
    return [
        {
            "entities": [
                {
                    "local_id": "tech",
                    "label": "Carbon conversion",
                    "kind": "Technology",
                    "evidence": [
                        {"chunk_id": "c1", "quote": "Carbon conversion"}
                    ],
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
                    "claim_id": "k",
                    "predicate": "solves_task",
                    "roles": {"subject": "tech", "task": "task"},
                    "polarity": "affirmed",
                    "modality": "reported",
                    "evidence": [{"chunk_id": "c1", "quote": TEXT}],
                }
            ],
            "context_requests": [],
        },
        {
            "items": [
                {"claim_id": "k", "decision": "supported", "reason": "Said."}
            ]
        },
    ]


REGISTRY = [
    Concept(
        concept_id="tech:capture",
        kind=ConceptKind.TECHNOLOGY,
        preferred_label="carbon capture",
        status="accepted",
    )
]


def run(deduplicator):
    return asyncio.run(
        process_document(
            document(),
            ReplayProvider(answers()),
            REGISTRY,
            PipelineSettings(
                primary_chunks=1,
                max_context_chunks=2,
                max_retries=0,
                retry_delay_seconds=0,
                max_retry_delay_seconds=0,
            ),
            semantic=deduplicator,
        )
    )


def test_semantic_match_is_a_review_candidate_and_keeps_the_claim():
    result = run(semantic())
    assert result.run.status == "succeeded"
    assert [item.status for item in result.assertions] == ["accepted"]
    decision = next(
        item
        for item in result.resolutions
        if item.method == SEMANTIC_CANDIDATE_METHOD
    )
    assert decision.status == "provisional"
    assert decision.concept_id != "tech:capture"
    assert decision.candidates[0]["concept_id"] == "tech:capture"
    assert decision.candidates[0]["kind"] == "Technology"
    assert result.embedding_model == "EmbeddingsGigaR"
    assert set(result.concept_embeddings) == {
        concept.concept_id for concept in result.concepts
    }
    assert result.run.metadata["semantic"]["status"] == "ok"
    assert result.run.metadata["semantic"]["candidates"] == 1

    tx = Transaction()
    asyncio.run(GraphStore._write_extraction(tx, document(), result))
    queries = [query for query, _ in tx.queries]
    assert any("POSSIBLY_SAME_AS" in query for query in queries)
    stored = next(
        parameters
        for query, parameters in tx.queries
        if "c.embedding = row.vector" in query
    )
    assert stored["model"] == "EmbeddingsGigaR"


def test_unavailable_embeddings_fall_back_to_lexical_resolution():
    deduplicator = semantic(fail=True)
    result = run(deduplicator)
    assert result.run.status == "succeeded"
    assert [item.status for item in result.assertions] == ["accepted"]
    assert not any(
        item.method == SEMANTIC_CANDIDATE_METHOD for item in result.resolutions
    )
    assert result.concept_embeddings == {}
    assert result.run.metadata["semantic"]["status"] == "unavailable"
    # One failed request pauses the layer instead of retrying per mention.
    assert deduplicator._embedder.calls == 1


def test_llm_runs_share_one_semantic_layer_unless_disabled(monkeypatch):
    from lctrend.extraction import processing

    created = []

    def factory():
        created.append(object())
        return created[-1]

    monkeypatch.setattr(processing, "_semantic_deduplicator", factory)
    monkeypatch.setattr(processing, "_SHARED_SEMANTIC", {})
    monkeypatch.setenv("DEDUP_IN_LLM", "1")
    assert processing._llm_semantic() is processing._llm_semantic()
    assert len(created) == 1
    monkeypatch.setenv("DEDUP_IN_LLM", "0")
    assert processing._llm_semantic() is None
