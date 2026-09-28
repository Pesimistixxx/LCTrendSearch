"""Bulk ingestion: concurrency, shared registry and paid-call safety."""

import asyncio
import sys
from pathlib import Path
from threading import Barrier

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm"))

from test_jobs import extraction, files, finish, manager  # noqa: E402
from test_llm_pipeline import document, settings  # noqa: E402

from lctrend.core.models import ConceptKind, Mention  # noqa: E402
from lctrend.extraction.resolver import (  # noqa: E402
    ConceptIndex,
    ConceptRegistry,
    resolve_mentions,
)
from lctrend.ingest.adapters import parse_openalex, parse_pypi  # noqa: E402
from lctrend.llm.client import JsonLLM, _record  # noqa: E402
from lctrend.llm.pipeline import process_document  # noqa: E402


def mention(mention_id, text, kind=ConceptKind.TECHNOLOGY):
    return Mention(
        mention_id=mention_id,
        chunk_id="c1",
        surface_text=text,
        start=0,
        end=len(text),
        type_candidates=[kind],
    )


def test_index_matches_names_synonyms_and_kind_like_a_full_scan():
    _, decisions = resolve_mentions(
        [
            mention("m1", "Graph Neural Network"),
            mention("m2", "graph neural networks"),
            mention("m3", "Graph neural network", ConceptKind.TASK),
        ],
        [],
    )
    # One provisional technology for both surface forms, a separate task.
    assert decisions[0].concept_id == decisions[1].concept_id
    assert decisions[2].concept_id != decisions[0].concept_id
    index = ConceptIndex()
    concepts, _ = resolve_mentions([mention("m4", "NLP")], index)
    assert len(index) == 1
    _, again = resolve_mentions(
        [mention("m5", "natural language processing")], index
    )
    assert again[0].concept_id == concepts[0].concept_id


def test_concurrent_documents_share_new_concepts_through_the_registry():
    registry = ConceptRegistry()

    async def both():
        return await asyncio.gather(
            registry.resolve([mention("a", "Sparse attention")]),
            registry.resolve([mention("b", "sparse attention")]),
        )

    (_, first), (_, second) = asyncio.run(both())
    assert first[0].concept_id == second[0].concept_id
    assert {first[0].status, second[0].status} == {"provisional", "accepted"}


def test_refetching_an_unchanged_record_keeps_its_version():
    work = {"id": "https://openalex.org/W1", "title": "Same work"}
    first = parse_openalex({**work, "_retrieved_at": "2026-01-01T00:00:00Z"})
    second = parse_openalex({**work, "_retrieved_at": "2026-02-01T00:00:00Z"})
    assert first.document_version_id == second.document_version_id
    # A new citation count is a metric observation, not new content (A-1).
    cited = parse_openalex({**work, "cited_by_count": 5})
    assert cited.document_version_id == first.document_version_id
    changed = parse_openalex({**work, "title": "Revised work"})
    assert changed.document_version_id != first.document_version_id
    package = {"info": {"name": "pkg", "version": "1.0"}, "releases": {}}
    assert (
        parse_pypi({**package, "_retrieved_at": "a"}).document_version_id
        == parse_pypi({**package, "_retrieved_at": "b"}).document_version_id
    )


def test_documents_of_one_job_are_processed_concurrently(tmp_path):
    barrier = Barrier(2, timeout=5)

    def process(doc, **kwargs):
        # Passes only when both documents are in flight at the same time.
        barrier.wait()
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    try:
        final = finish(
            instance,
            instance.create_files(files(tmp_path), workers=2),
        )
        assert final["counts"]["succeeded"] == 2
    finally:
        instance.close(wait=True)


def test_registry_is_read_once_per_job(tmp_path):
    instance = manager(tmp_path)
    try:
        finish(instance, instance.create_files(files(tmp_path, 3)))
        assert instance.fixture_store.events.count("registry") == 1
    finally:
        instance.close(wait=True)


def test_already_processed_version_is_skipped_without_model_calls(tmp_path):
    processed = []

    def process(doc, **kwargs):
        processed.append(doc.document_id)
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    store = instance.fixture_store
    store.processed_versions = lambda ids: {ids[0]} if ids else set()
    try:
        final = finish(instance, instance.create_files(files(tmp_path, 1)))
        doc = final["documents"][0]
        assert processed == [] and store.writes == []
        assert doc["status"] == "succeeded"
        assert doc["stage"] == "already_processed"
        assert doc["llm_status"] == "skipped"
        skipped = instance.get_result(final["job_id"], doc["doc_id"])
        assert skipped["extraction"] is None
    finally:
        instance.close(wait=True)


class InterleavingProvider:
    """One shared provider; each call yields so documents interleave."""

    demo = False
    models = {"extract": "offline", "review": "offline"}

    def __init__(self):
        self.calls = []

    async def generate(self, schema, system, payload, *, stage="extract"):
        await asyncio.sleep(0)
        _record(self.calls, {"stage": stage})
        await asyncio.sleep(0)
        return schema.model_validate({"entities": [], "claims": []})


def test_concurrent_documents_audit_only_their_own_provider_calls():
    provider = InterleavingProvider()
    first = document(["Alpha sensor."])
    second = document(["Beta sensor.", "Gamma sensor."])

    async def both():
        return await asyncio.gather(
            process_document(first, provider, settings=settings()),
            process_document(second, provider, settings=settings()),
        )

    one, two = asyncio.run(both())
    assert len(provider.calls) == 3
    assert len(one.run.metadata["provider_calls"]) == 1
    assert len(two.run.metadata["provider_calls"]) == 2


def test_provider_concurrency_limit_bounds_in_flight_requests(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    active, peak = 0, 0

    async def respond(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"entities": []}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    provider = JsonLLM(
        "model",
        base_url="http://127.0.0.1:1/v1",
        transport=httpx.MockTransport(respond),
    )
    from lctrend.llm.contracts import Extraction

    async def many():
        await asyncio.gather(
            *(provider.generate(Extraction, "s", {}) for _ in range(6))
        )

    asyncio.run(many())
    assert peak == 2
    assert len(provider.calls) == 6
