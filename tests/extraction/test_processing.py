"""CLI/web parity through the real extraction pipeline, without services."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from frontend.server.jobs import JobManager
from lctrend import cli
from lctrend.core.models import (
    Artifact,
    Chunk,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.extraction import processing
from lctrend.extraction.processing import NerRuntime, process_material
from lctrend.llm.client import LLMError, ReplayProvider


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


class Ner:
    def predict_entities(self, text, labels, threshold):
        return [
            {"start": 0, "end": 8, "label": "technology", "score": 0.95},
            {"start": 16, "end": 26, "label": "task", "score": 0.92},
        ]


class Store:
    def __init__(self):
        self.results = []

    def ensure_schema(self):
        pass

    def read_concepts(self):
        return []

    def write_processed(self, doc, result):
        self.results.append(result)

    def write_document(self, doc):
        self.results.append(None)

    def close(self):
        pass


@pytest.mark.parametrize("mode", ["hybrid", "llm", "gliner", "none"])
@pytest.mark.parametrize("ner_failure", [False, True])
def test_web_and_cli_share_extraction_contract(
    tmp_path, monkeypatch, mode, ner_failure
):
    class BrokenNer:
        def predict_entities(self, *args, **kwargs):
            raise RuntimeError("offline inference failure")

    # GLiNER-only inference errors propagate to the caller, unlike hybrid.
    if mode == "gliner" and ner_failure:
        with pytest.raises(RuntimeError):
            asyncio.run(
                process_material(document(), mode=mode, ner_model=BrokenNer())
            )
        return
    model = BrokenNer() if ner_failure else Ner()
    monkeypatch.setattr(cli, "_semantic_deduplicator", lambda: None)
    monkeypatch.setattr(processing, "_semantic_deduplicator", lambda: None)
    cli_store, web_store = Store(), Store()
    cli._write_ingested(
        document(),
        mode != "none",
        "offline",
        cli_store,
        model=model,
        extractor=mode,
        provider=ReplayProvider(responses()),
    )
    path = tmp_path / "source.txt"
    path.write_text("fixture", encoding="utf-8")
    manager = JobManager(
        tmp_path / "jobs",
        store_factory=lambda: web_store,
        file_parser=lambda path: document(),
        ner_runtime_factory=lambda: NerRuntime(model, "offline"),
        provider_factory=lambda: ReplayProvider(responses()),
    )
    try:
        job = manager.create_files([path], mode=mode)
        manager._futures[job["job_id"]].result(timeout=10)
        final = manager.get_job(job["job_id"])
        assert final["counts"]["succeeded"] == 1
        expected, actual = cli_store.results[0], web_store.results[0]
        if mode == "none":
            assert expected is actual is None
            return
        assert actual.model_dump(exclude={"run"}) == expected.model_dump(
            exclude={"run"}
        )
        assert actual.run.parser == expected.run.parser
        assert actual.run.status == expected.run.status
        assert actual.run.metadata.get("ner") == expected.run.metadata.get(
            "ner"
        )
        assert "parallel_gliner" not in actual.run.metadata
        saved = manager.get_result(
            job["job_id"], final["documents"][0]["doc_id"]
        )
        assert saved["extraction"] == actual.model_dump(mode="json")
        if mode == "hybrid":
            assert actual.run.metadata["ner"]["status"] == (
                "failed" if ner_failure else "ok"
            )
            assert actual.assertions[0].status == "accepted"
            if not ner_failure:
                assert actual.mentions[0].confidence is not None
    finally:
        manager.close(wait=True)


def test_failed_llm_has_no_verified_ner_fallback():
    class BrokenProvider:
        def generate(self, *args, **kwargs):
            raise LLMError("auth", "Credentials rejected", retryable=False)

    result = asyncio.run(
        process_material(
            document(), provider=BrokenProvider(), ner_model=Ner()
        )
    )
    assert result.run.status == "failed"
    assert result.assertions == []
    assert all(m.mention_role == "ner_candidate" for m in result.mentions)


def test_llm_mode_does_not_call_ner():
    class ForbiddenNer:
        def predict_entities(self, *args, **kwargs):
            pytest.fail("LLM-only must not call NER")

    result = asyncio.run(
        process_material(
            document(),
            mode="llm",
            provider=ReplayProvider(responses()),
            ner_model=ForbiddenNer(),
        )
    )
    assert result.run.metadata["ner"]["status"] == "disabled"


def test_invalid_mode_fails_before_starting_work():
    with pytest.raises(ValueError, match="mode must"):
        asyncio.run(process_material(document(), mode="search"))


def test_shared_runtime_serializes_ner_forward_calls():
    started, release = Event(), Event()
    guard = Lock()
    active = peak = calls = 0

    class SharedModel:
        def predict_entities(self, *args, **kwargs):
            nonlocal active, peak, calls
            with guard:
                active += 1
                peak = max(peak, active)
                calls += 1
            started.set()
            assert release.wait(timeout=3)
            with guard:
                active -= 1
            return []

    runtime = NerRuntime(SharedModel(), model_name="offline")
    assert runtime.prepare() is runtime
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runtime.predict_entities, "text", [])
        assert started.wait(timeout=3)
        second = pool.submit(runtime.predict_entities, "text", [])
        release.set()
        first.result(timeout=3)
        second.result(timeout=3)
    assert peak == 1
    assert calls == 2
