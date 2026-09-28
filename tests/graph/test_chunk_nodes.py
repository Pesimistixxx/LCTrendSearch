"""Only chunks something stands on become graph nodes."""

import asyncio

from lctrend.core.models import (
    Artifact,
    Assertion,
    Chunk,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    EvidenceSpan,
    ExtractionResult,
    Mention,
    ProcessingRun,
    SourceRef,
)
from lctrend.graph.store import GraphStore, evidence_chunk_ids


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
            Chunk(
                chunk_id=f"c{index}", kind="fulltext", text=text, order=index
            )
            for index, text in enumerate(
                [
                    "Sensor S detects leaks.",
                    "We surveyed 40 plants.",
                    "Tables.",
                ]
            )
        ],
    )


def result():
    return ExtractionResult(
        document_version_id="v1",
        run=ProcessingRun(
            run_id="run", parser="llm_packets", config_hash="x", started_at="t"
        ),
        mentions=[
            Mention(
                mention_id="m",
                chunk_id="c0",
                surface_text="Sensor S",
                start=0,
                end=8,
                type_candidates=[ConceptKind.TECHNOLOGY],
            )
        ],
        assertions=[
            Assertion(
                assertion_id="a",
                predicate="solves_task",
                roles={},
                evidence=[
                    EvidenceSpan(
                        chunk_id="c1", quote="We surveyed", start=0, end=11
                    )
                ],
            )
        ],
    )


def test_mentions_and_quotes_keep_their_chunks_and_nothing_else():
    assert evidence_chunk_ids(result()) == {"c0", "c1"}
    # A document written without extraction keeps no chunk nodes.
    assert evidence_chunk_ids(None) == set()


def test_the_all_policy_keeps_every_chunk(monkeypatch):
    monkeypatch.setattr(
        "lctrend.graph.store.load_catalog",
        lambda name: {"chunk_nodes": "all"},
    )
    assert evidence_chunk_ids(result()) is None


class Transaction:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        return self

    def consume(self):
        return None


def test_only_kept_chunks_are_written_and_linked(monkeypatch):
    async def nothing(*args, **kwargs):
        return None

    # The work layer (graph.works) reads its keys back; not under test here.
    monkeypatch.setattr(GraphStore, "_write_work", staticmethod(nothing))
    tx = Transaction()
    asyncio.run(
        GraphStore._write_document(
            tx, document(), publishing=True, kept_chunk_ids={"c0"}
        )
    )

    written = [
        row["chunk_id"]
        for query, parameters in tx.queries
        if "MERGE (c:Chunk" in query
        for row in parameters["rows"]
    ]
    assert written == ["c0"]
    # Chunks leaving the version are unlinked by the kept set.
    unlinked = [
        parameters["chunk_ids"]
        for query, parameters in tx.queries
        if "DELETE active" in query
    ]
    assert unlinked == [["c0"]]


class PruneSession:
    def __init__(self, found, batches):
        self.found = found
        self.batches = list(batches)
        self.queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def run(self, query, **parameters):
        self.queries.append(query)
        if "RETURN count(c) AS chunks" in query:
            return [self.found]
        if "RETURN c.chunk_id AS chunk_id" in query:
            return [
                {"chunk_id": f"c{index}"}
                for index in range(self.found["chunks"])
            ]
        return self

    def consume(self):
        return None

    def single(self):
        return {"deleted": self.batches.pop(0)}

    async def execute_write(self, callback, *args):
        return await callback(self, *args)


def prune(session, **options):
    store = object.__new__(GraphStore)
    store._database = None
    store._driver = type(
        "Driver", (), {"session": lambda self, **_: session}
    )()
    return asyncio.run(store.prune_text_chunks(**options))


def test_a_dry_run_only_counts():
    session = PruneSession({"chunks": 1251, "chars": 266000}, [])
    summary = prune(session)

    assert summary["chunks"] == 1251 and summary["deleted"] == 0
    assert "Irreversible" in summary["warning"]
    assert not any("DELETE" in query for query in session.queries)


def test_apply_records_run_inputs_then_deletes_in_batches():
    session = PruneSession({"chunks": 1500, "chars": 1}, [1000, 500])
    summary = prune(session, apply=True, batch=1000)

    assert summary["deleted"] == 1500
    inputs = next(
        i for i, q in enumerate(session.queries) if "input_chunk_ids" in q
    )
    first_delete = next(
        i for i, q in enumerate(session.queries) if "DETACH DELETE" in q
    )
    assert inputs < first_delete
