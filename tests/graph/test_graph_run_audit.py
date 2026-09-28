"""Verify Neo4j-compatible audit projection without a database connection."""

import asyncio
import json

import pytest

from lctrend.core.models import (
    Assertion,
    Concept,
    ConceptKind,
    EvidenceSpan,
    ExtractionResult,
    Mention,
    ProcessingRun,
    ResolutionDecision,
    validate_extraction,
)
from lctrend.graph.store import GraphStore
from lctrend.ingest.adapters import parse_openalex


class Result:
    def __init__(self, count=0, published=0):
        self.count = count
        self.published = published

    def consume(self):
        return None

    def single(self):
        return {"count": self.count, "published": self.published}


class Transaction:
    def __init__(self, existing_count=0, published=0):
        self.queries = []
        self.existing_count = existing_count
        self.published = published

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return Result(self.existing_count, self.published)


def project(run):
    document = parse_openalex(
        {"id": "https://openalex.org/W123", "title": "Audited fixture"}
    )
    tx = Transaction()
    extraction = ExtractionResult(
        document_version_id=document.document_version_id, run=run
    )
    asyncio.run(GraphStore._write_extraction(tx, document, extraction))
    query, parameters = next(
        (query, parameters)
        for query, parameters in tx.queries
        if "MERGE (r:ProcessingRun" in query
    )
    return document, tx, query, parameters


def test_nested_llm_run_audit_is_serialized_to_json_preserving_native_fields():
    metadata = {
        "demo": False,
        "source_truth_assessed": False,
        "coverage": {
            "processed_focus_chunk_ids": ["ч1"],
            "unprocessed_chunk_ids": [],
        },
        "issues": [{"code": "unresolved_context", "details": {"limit": 2}}],
        "provider_calls": [
            {
                "stage": "extract",
                "cache_hit": True,
                "tokens": {},
                "estimated_cost_usd": None,
            }
        ],
    }
    trace = [
        {
            "stage": "plan",
            "packets": [{"packet_id": "p1", "focus_chunk_ids": ["ч1"]}],
        },
        {
            "stage": "verification",
            "items": [
                {
                    "claim_id": "c1",
                    "reason": "Цитата подтверждена",
                    "decision": "supported",
                }
            ],
        },
    ]
    run = ProcessingRun(
        run_id="run:audit",
        pipeline_version="material-llm/1",
        parser="llm_packets",
        model_revision='{"extract":"small","review":"large"}',
        prompt_hash="prompt:hash",
        config_hash="config:hash",
        started_at="2026-09-26T00:00:00+00:00",
        status="succeeded",
        metadata=metadata,
        trace=trace,
    )
    before = run.model_dump(mode="json")
    document, tx, query, parameters = project(run)
    assert "r.metadata_json = $metadata_json" in query
    assert "r.trace_json = $trace_json" in query
    assert "metadata" not in parameters and "trace" not in parameters
    assert isinstance(parameters["metadata_json"], str) and isinstance(
        parameters["trace_json"], str
    )
    assert json.loads(parameters["metadata_json"]) == {
        **metadata,
        "publication_status": "published",
    }
    assert json.loads(parameters["trace_json"]) == trace
    assert "Цитата" in parameters["trace_json"]
    assert parameters["version_id"] == document.document_version_id
    for key, value in run.model_dump(exclude={"metadata", "trace"}).items():
        assert parameters[key] == value
        assert "$" + key in query
    assert run.model_dump(mode="json") == before
    assert any(
        "prior.status = 'superseded'" in statement
        and values["run_id"] == run.run_id
        for statement, values in tx.queries
    )


def test_legacy_run_baseline_fields_and_empty_audit_are_preserved():
    run = ProcessingRun(
        run_id="run:legacy",
        parser="llm",
        model_revision="original-ner-model",
        config_hash="legacy-config",
        started_at="2026-01-01T00:00:00Z",
    )
    _, _, query, parameters = project(run)
    assert parameters["pipeline_version"] == "0.1.0"
    assert parameters["parser"] == "llm"
    assert parameters["model_revision"] == "original-ner-model"
    assert parameters["prompt_hash"] is None
    assert parameters["config_hash"] == "legacy-config"
    assert parameters["status"] == "succeeded"
    assert json.loads(parameters["metadata_json"]) == {
        "publication_status": "published"
    }
    assert json.loads(parameters["trace_json"]) == []
    assert "MERGE (r)-[:PROCESSED]->(v)" in query


@pytest.mark.parametrize(
    "status,publication_status", [("failed", "failed"), ("partial", "staged")]
)
def test_incomplete_run_preserves_prior_active_projection(
    status, publication_status
):
    document = parse_openalex(
        {
            "id": "https://openalex.org/W9",
            "title": "Staged fixture",
            "abstract_inverted_index": {
                "Sensor": [0],
                "S": [1],
                "has": [2],
                "limitations.": [3],
            },
        }
    )
    run = ProcessingRun(
        run_id="run:incomplete",
        parser="llm_packets",
        config_hash="config:bounded",
        started_at="2026-09-26T00:00:00Z",
        status=status,
        metadata={
            "coverage": {"unprocessed_chunk_ids": ["c:omitted"]},
            "model_calls": 2,
        },
        trace=[{"stage": "review", "status": "failed", "code": "timeout"}],
    )
    extraction = ExtractionResult(
        document_version_id=document.document_version_id, run=run
    )
    if status == "partial":
        chunk = document.chunks[0]
        extraction.concepts = [
            Concept(
                concept_id="technology:staged",
                preferred_label="Sensor S",
                kind=ConceptKind.TECHNOLOGY,
            )
        ]
        extraction.mentions = [
            Mention(
                mention_id="mention:staged",
                chunk_id=chunk.chunk_id,
                surface_text="Sensor S",
                start=0,
                end=8,
                type_candidates=[ConceptKind.TECHNOLOGY],
            )
        ]
        extraction.resolutions = [
            ResolutionDecision(
                resolution_id="resolution:staged",
                mention_id="mention:staged",
                concept_id="technology:staged",
                status="provisional",
            )
        ]
        extraction.assertions = [
            Assertion(
                assertion_id="assertion:staged",
                predicate="reports_limitation",
                roles={"subject": "technology:staged"},
                evidence=[
                    EvidenceSpan(
                        chunk_id=chunk.chunk_id,
                        quote=chunk.text,
                        start=0,
                        end=len(chunk.text),
                    )
                ],
                qualifiers={"conditions": {"temperature": 25}},
                values=[],
                status="needs_review",
                verification_status="unverified",
            )
        ]
    validate_extraction(document, extraction)
    tx = Transaction()
    asyncio.run(GraphStore._write_extraction(tx, document, extraction))
    assert len(tx.queries) == 1
    query, parameters = tx.queries[0]
    assert "MERGE (r:ProcessingRun" in query
    assert "superseded" not in query
    assert "DELETE" not in query
    assert (
        "MENTIONS" not in query
        and "SOLVES" not in query
        and "HAS_ECONOMIC_EVIDENCE" not in query
    )
    assert (
        "MERGE (a:Assertion" not in query
        and "MERGE (c:Technology" not in query
    )
    assert parameters["status"] == status
    stored_metadata = json.loads(parameters["metadata_json"])
    assert stored_metadata["publication_status"] == publication_status
    assert stored_metadata["coverage"] == run.metadata["coverage"]
    assert json.loads(parameters["trace_json"]) == run.trace
    staged = stored_metadata["staged_result"]
    # Label vectors are recomputable and would bloat the run audit.
    assert staged == extraction.model_dump(
        mode="json", exclude={"run", "concept_embeddings"}
    )
    assert stored_metadata["staged_chunks"] == [
        chunk.model_dump(mode="json") for chunk in document.chunks
    ]
    if status == "partial":
        # The document-local validated candidate remains recoverable for a
        # later review, even though it has not replaced the active graph
        # projection.
        recovered = ExtractionResult.model_validate(
            {**staged, "run": run.model_dump(mode="json")}
        )
        validate_extraction(document, recovered)
        assert recovered.assertions[0].qualifiers == {
            "conditions": {"temperature": 25}
        }
        assert (
            recovered.assertions[0].evidence[0].quote
            == document.chunks[0].text
        )


@pytest.mark.parametrize(
    "status,existing_count,published,expected,publish",
    [
        ("failed", 1, 0, ["extraction"], False),
        # A partial rerun never replaces an active projection...
        ("partial", 1, 1, ["extraction"], False),
        # ...but is published when the version has none yet.
        ("partial", 1, 0, ["document", "extraction"], True),
        ("partial", 0, 0, ["document", "extraction"], True),
        ("failed", 0, 0, ["document", "extraction"], False),
        ("succeeded", 1, 1, ["document", "extraction"], True),
        ("succeeded", 0, 0, ["document", "extraction"], True),
    ],
)
def test_processed_write_keeps_existing_chunks_on_incomplete_rerun(
    monkeypatch, status, existing_count, published, expected, publish
):
    document = parse_openalex(
        {"id": "https://openalex.org/W88", "title": "Transaction fixture"}
    )
    result = ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id="run:transaction",
            parser="llm_packets",
            config_hash="config",
            started_at="2026-09-26T00:00:00Z",
            status=status,
        ),
    )
    tx = Transaction(existing_count, published)
    calls = []
    monkeypatch.setattr(
        GraphStore,
        "_write_document",
        staticmethod(
            lambda transaction, doc, publishing=False: calls.append(
                ("document", transaction, doc)
            )
        ),
    )
    monkeypatch.setattr(
        GraphStore,
        "_write_extraction",
        staticmethod(
            lambda transaction, doc, extraction, publish=None: calls.append(
                ("extraction", transaction, doc, extraction, publish)
            )
        ),
    )
    asyncio.run(GraphStore._write_processed(tx, document, result))
    assert [call[0] for call in calls] == expected
    assert all(call[1] is tx and call[2] is document for call in calls)
    assert calls[-1][3] is result
    assert calls[-1][4] is publish
    assert len(tx.queries) == 1
    assert "RETURN count(v) AS count" in tx.queries[0][0]
    assert tx.queries[0][1]["version_id"] == document.document_version_id
    # D-3: the run being published does not count as an active better one.
    assert "r.run_id <> $run_id" in tx.queries[0][0]
    assert tx.queries[0][1]["run_id"] == result.run.run_id


def test_document_and_extraction_share_one_execute_write_transaction(
    monkeypatch,
):
    document = parse_openalex(
        {"id": "https://openalex.org/W99", "title": "Atomic fixture"}
    )
    result = ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id="run:atomic",
            parser="llm_packets",
            config_hash="config",
            started_at="2026-09-26T00:00:00Z",
        ),
    )
    tx = Transaction(existing_count=1)
    writes = []
    transactions = []

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def execute_write(self, callback, *args):
            transactions.append(callback)
            return callback(tx, *args)

    class Driver:
        def session(self, **kwargs):
            assert kwargs == {"database": "offline"}
            return Session()

    store = object.__new__(GraphStore)
    store._driver = Driver()
    store._database = "offline"
    monkeypatch.setattr(
        GraphStore,
        "_write_document",
        staticmethod(
            lambda transaction, doc, publishing=False: writes.append(
                ("document", transaction)
            )
        ),
    )
    monkeypatch.setattr(
        GraphStore,
        "_write_extraction",
        staticmethod(
            lambda transaction, doc, result, publish=None: writes.append(
                ("extraction", transaction)
            )
        ),
    )
    asyncio.run(store.write_processed(document, result))
    assert len(transactions) == 1
    assert transactions[0] == GraphStore._write_processed
    assert writes == [("document", tx), ("extraction", tx)]
    result.document_version_id = "another-document-version"
    with pytest.raises(ValueError, match="another document version"):
        asyncio.run(store.write_processed(document, result))
    assert len(transactions) == 1


def test_successful_empty_extraction_replaces_only_active_assertion_links():
    run = ProcessingRun(
        run_id="run:empty-success",
        parser="llm_packets",
        config_hash="config",
        started_at="2026-09-26T00:00:00Z",
    )
    document, tx, _, _ = project(run)
    deletes = [
        (query, parameters)
        for query, parameters in tx.queries
        if "DELETE" in query and "HAS_ASSERTION" in query
    ]
    assert len(deletes) == 1
    query, parameters = deletes[0]
    assert parameters["version_id"] == document.document_version_id
    assert "DETACH DELETE" not in query
    assert "CREATED" not in query
    # Removing an active document link must leave the assertion node and its
    # historical run links available, even when the new response has no claims.
    assert not any(
        "DELETE" in statement and "CREATED" in statement
        for statement, _ in tx.queries
    )
    assert not any(
        "DETACH DELETE" in statement and "Assertion" in statement
        for statement, _ in tx.queries
    )


def test_review_history_is_saved_on_each_run_creation_link():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W7",
            "title": "History",
            "abstract_inverted_index": {
                "Sensor": [0],
                "S": [1],
                "has": [2],
                "limitations.": [3],
            },
        }
    )
    chunk = document.chunks[0]
    records = []
    for run_id, status, verification in (
        ("run:older", "accepted", "supported"),
        ("run:newer", "rejected", "unsupported"),
    ):
        run = ProcessingRun(
            run_id=run_id,
            parser="llm_packets",
            config_hash="config",
            started_at="2026-09-26T00:00:00Z",
        )
        extraction = ExtractionResult(
            document_version_id=document.document_version_id,
            run=run,
            concepts=[
                Concept(
                    concept_id="tech:shared",
                    preferred_label="Sensor S",
                    kind=ConceptKind.TECHNOLOGY,
                )
            ],
            assertions=[
                Assertion(
                    assertion_id="assertion:shared",
                    predicate="reports_limitation",
                    roles={"subject": "tech:shared"},
                    status=status,
                    verification_status=verification,
                    evidence=[
                        EvidenceSpan(
                            chunk_id=chunk.chunk_id,
                            quote=chunk.text,
                            start=0,
                            end=len(chunk.text),
                        )
                    ],
                )
            ],
        )
        validate_extraction(document, extraction)
        tx = Transaction()
        asyncio.run(GraphStore._write_extraction(tx, document, extraction))
        query, parameters = next(
            (statement, values)
            for statement, values in tx.queries
            if "MERGE (a:Assertion" in statement
        )
        assert "MERGE (r)-[creation:CREATED]->(a)" in query
        assert "creation.status = row.status" in query
        assert (
            "creation.verification_status = row.verification_status" in query
        )
        (row,) = parameters["rows"]
        records.append(
            (
                parameters["run_id"],
                row["assertion_id"],
                row["status"],
                row["verification_status"],
            )
        )
    assert records == [
        ("run:older", "assertion:shared", "accepted", "supported"),
        ("run:newer", "assertion:shared", "rejected", "unsupported"),
    ]
