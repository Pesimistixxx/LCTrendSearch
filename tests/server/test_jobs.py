"""Offline checks of persistence, cancellation, provenance and bounded
workers.
"""

import json
from copy import deepcopy
from pathlib import Path
from threading import Barrier, Event, Lock
from time import sleep

import pytest

from frontend.server.jobs import JobManager
from lctrend.core.models import (
    Artifact,
    Chunk,
    DocumentEnvelope,
    DocumentType,
    ExtractionResult,
    ProcessingRun,
    SourceRef,
)


def document(path: Path):
    return DocumentEnvelope(
        document_id=path.stem,
        document_version_id=f"v:{path.stem}",
        document_type=DocumentType.REPORT,
        title=path.name,
        source=SourceRef(
            source_id="fixture",
            name="Fixture",
            source_type="test",
            record_id=path.name,
        ),
        artifact=Artifact(
            uri="memory://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        coverage="full_text",
        chunks=[
            Chunk(
                chunk_id="c1",
                kind="paragraph",
                text="Sensor S solves monitoring.",
                order=0,
            )
        ],
    )


def extraction(doc, status="succeeded", **metadata):
    return ExtractionResult(
        document_version_id=doc.document_version_id,
        run=ProcessingRun(
            run_id=f"run:{doc.document_id}",
            parser="llm",
            config_hash="fixture",
            started_at="2026-09-26T00:00:00+00:00",
            status=status,
            metadata=metadata,
        ),
    )


class Store:
    def __init__(self):
        self.events = []
        self.writes = []
        self.closed = False

    def verify_connectivity(self):
        self.events.append("verify")

    def ensure_schema(self):
        self.events.append("schema")

    def read_concepts(self):
        self.events.append("registry")
        return []

    def write_processed(self, doc, result):
        self.writes.append((doc, result))

    def write_document(self, doc):
        self.writes.append((doc, None))

    def close(self):
        self.closed = True


def files(tmp_path, count=2):
    paths = [tmp_path / f"study-{index}.txt" for index in range(count)]
    for path in paths:
        path.write_text("Sensor S solves monitoring.", encoding="utf-8")
    return paths


def manager(tmp_path, **overrides):
    store = Store()
    defaults = dict(
        store_factory=lambda: store,
        file_parser=document,
        provider_factory=object,
        document_processor=lambda doc, **kwargs: extraction(doc),
        snapshot_writer=lambda doc, raw: doc,
        pdf_support_checker=lambda: None,
    )
    defaults.update(overrides)
    instance = JobManager(tmp_path / "jobs", **defaults)
    instance.fixture_store = store
    return instance


def finish(instance, job):
    instance._futures[job["job_id"]].result(timeout=5)
    return instance.get_job(job["job_id"])


def test_files_have_truthful_progress_and_persistent_results(tmp_path):
    started, release = Event(), Event()
    providers = []

    def process(doc, event, provider, **kwargs):
        providers.append(provider)
        event(
            {
                "branch": "llm",
                "stage": "review",
                "status": "running",
                "packet_id": "packet:1",
            }
        )
        started.set()
        assert release.wait(3)
        return extraction(
            doc,
            "partial",
            coverage={"total_chunks": 1, "unprocessed_chunk_ids": ["c1"]},
        )

    instance = manager(tmp_path, document_processor=process)
    try:
        job = instance.create_files(files(tmp_path), direction="Sensors")
        assert started.wait(3)
        active = instance.get_job(job["job_id"])
        assert active["counts"] == {
            "total": 2,
            "discovered": 2,
            "completed": 0,
            "queued": 1,
            "running": 1,
            "succeeded": 0,
            "partial": 0,
            "failed": 0,
            "cancelled": 0,
        }
        assert active["documents"][0]["stage"] == "review"
        release.set()
        final = finish(instance, job)
        assert final["status"] == "completed"
        assert final["counts"]["partial"] == 2
        assert final["documents"][0]["llm_status"] == "partial"
        # One provider per job: one token, one rate limit, one ladder.
        assert providers[0] is providers[1]
        assert instance.fixture_store.events[:2] == ["verify", "schema"]
        result = instance.get_result(
            job["job_id"], final["documents"][0]["doc_id"]
        )
        assert result["document"]["coverage"] == "full_text"
        assert result["extraction"]["run"]["status"] == "partial"
        assert (
            json.loads(
                (instance.directory / job["job_id"] / "job.json").read_text(
                    encoding="utf-8"
                )
            )
            == final
        )
        assert instance.fixture_store.closed
    finally:
        release.set()
        instance.close(wait=True)


def test_document_without_text_reports_why_llm_was_not_called(tmp_path):
    def parse(path):
        return document(path).model_copy(
            update={"chunks": [], "coverage": "metadata_only"}
        )

    instance = manager(
        tmp_path,
        file_parser=parse,
        document_processor=lambda doc, **kwargs: extraction(
            doc, "failed", model_calls=0, coverage={"total_chunks": 0}
        ),
    )
    try:
        final = finish(instance, instance.create_files(files(tmp_path, 1)))
        record = final["documents"][0]
        assert record["status"] == "failed"
        assert record["llm_status"] == "no_text"
        assert record["error"]["code"] == "no_text"
        assert "LLM не вызывалась" in record["error"]["message"]
        saved = instance.get_result(final["job_id"], record["doc_id"])
        assert saved["extraction"]["run"]["status"] == "failed"
        assert saved["document"]["chunks"] == []
    finally:
        instance.close(wait=True)


def test_cancel_finishes_current_document_and_never_starts_remaining(tmp_path):
    started, release = Event(), Event()
    seen = []

    def process(doc, **kwargs):
        seen.append(doc.title)
        started.set()
        assert release.wait(3)
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    try:
        job = instance.create_files(files(tmp_path, 3))
        assert started.wait(3)
        assert instance.cancel_job(job["job_id"])["status"] == "cancelling"
        release.set()
        final = finish(instance, job)
        assert final["status"] == "cancelled"
        assert final["counts"]["succeeded"] == 1
        assert final["counts"]["cancelled"] == 2
        assert len(instance.fixture_store.writes) == len(seen) == 1
        assert [doc["status"] for doc in final["documents"]] == [
            "succeeded",
            "cancelled",
            "cancelled",
        ]
    finally:
        release.set()
        instance.close(wait=True)


def test_cancelling_a_queued_job_never_opens_graph_or_calls_models(tmp_path):
    started, release = Event(), Event()
    stores = []

    def factory():
        store = Store()
        stores.append(store)
        return store

    def process(doc, **kwargs):
        started.set()
        assert release.wait(3)
        return extraction(doc)

    instance = manager(
        tmp_path, store_factory=factory, document_processor=process
    )
    try:
        paths = files(tmp_path)
        first = instance.create_files(paths[:1])
        assert started.wait(3)
        second = instance.create_files(paths[1:])
        final = instance.cancel_job(second["job_id"])
        assert final["status"] == "cancelled"
        assert final["counts"]["cancelled"] == 1
        assert len(stores) == 1
        release.set()
        finish(instance, first)
    finally:
        release.set()
        instance.close(wait=True)


def test_restart_marks_pending_jobs_interrupted_without_paid_auto_retry(
    tmp_path,
):
    instance = manager(tmp_path)
    job = instance.create_files(files(tmp_path))
    final = finish(instance, job)
    instance.close(wait=True)
    modified = deepcopy(final)
    modified["status"] = "running"
    modified["documents"][0]["status"] = "running"
    modified["documents"][1]["status"] = "queued"
    path = instance.directory / job["job_id"] / "job.json"
    path.write_text(json.dumps(modified), encoding="utf-8")

    def forbidden():
        raise AssertionError("restart must never invoke dependencies")

    restored = JobManager(
        instance.directory, store_factory=forbidden, provider_factory=forbidden
    )
    try:
        result = restored.get_job(job["job_id"])
        assert result["status"] == "interrupted"
        assert result["error"]["code"] == "interrupted"
        assert result["counts"]["failed"] == result["counts"]["cancelled"] == 1
        assert (
            restored.get_result(job["job_id"], "d000001")["extraction"]["run"][
                "status"
            ]
            == "succeeded"
        )
        assert restored._futures == {}
    finally:
        restored.close(wait=True)


def test_one_parse_failure_does_not_block_other_documents_or_leak_exception(
    tmp_path,
):
    paths = files(tmp_path)

    def parser(path):
        if path == paths[0]:
            raise ValueError("private API key = secret-data")
        return document(path)

    instance = manager(tmp_path, file_parser=parser)
    try:
        final = finish(instance, instance.create_files(paths))
        assert final["status"] == "completed"
        assert final["counts"]["failed"] == final["counts"]["succeeded"] == 1
        assert final["documents"][0]["llm_status"] == "not_started"
        assert "secret-data" not in json.dumps(final)
        assert len(instance.fixture_store.writes) == 1
    finally:
        instance.close(wait=True)


def test_neo4j_preflight_failure_happens_before_any_model_configuration(
    tmp_path,
):
    class Offline(Store):
        def verify_connectivity(self):
            raise RuntimeError("bolt://neo4j:password@database")

    def forbidden():
        raise AssertionError(
            "model must not be configured before graph is available"
        )

    instance = manager(
        tmp_path,
        store_factory=Offline,
        provider_factory=forbidden,
    )
    try:
        final = finish(instance, instance.create_files(files(tmp_path)))
        assert final["status"] == "failed"
        assert final["stage"] == "failed"
        assert "Neo4j" in final["error"]["message"]
        assert "password" not in json.dumps(final)
        assert final["counts"]["cancelled"] == 2
    finally:
        instance.close(wait=True)


def test_provider_configuration_failure_is_sanitized_before_document_work(
    tmp_path,
):
    def missing():
        raise RuntimeError("missing api key; actual secret = key123")

    instance = manager(tmp_path, provider_factory=missing)
    try:
        final = finish(instance, instance.create_files(files(tmp_path)))
        assert final["status"] == "failed"
        assert "LLM" in final["error"]["message"]
        assert "key123" not in json.dumps(final)
        assert instance.fixture_store.writes == []
    finally:
        instance.close(wait=True)


def test_graph_publication_failure_keeps_local_result_reviewable(tmp_path):
    class BrokenStore(Store):
        def write_processed(self, *args):
            raise RuntimeError("private database password")

    instance = manager(tmp_path, store_factory=BrokenStore)
    try:
        final = finish(instance, instance.create_files(files(tmp_path, 1)))
        doc = final["documents"][0]
        assert final["status"] == doc["status"] == "failed"
        assert doc["result_ready"]
        assert doc["stage"] == "publication"
        assert "локально" in doc["error"]["message"]
        assert (
            instance.get_result(final["job_id"], doc["doc_id"])["extraction"][
                "run"
            ]["status"]
            == "succeeded"
        )
        assert "password" not in json.dumps(final)
    finally:
        instance.close(wait=True)


def test_document_workers_are_bounded_and_graph_publication_is_serialized(
    tmp_path,
):
    barrier = Barrier(2)
    lock = Lock()
    active = peak = 0
    publication_active = publication_peak = 0

    def process(doc, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=3)
        with lock:
            active -= 1
        return extraction(doc)

    class SerialStore(Store):
        def write_processed(self, doc, result):
            nonlocal publication_active, publication_peak
            with lock:
                publication_active += 1
                publication_peak = max(publication_peak, publication_active)
            # Independent graph writes would overlap while waiting for network.
            sleep(0.02)
            try:
                super().write_processed(doc, result)
            finally:
                with lock:
                    publication_active -= 1

    manager_instance = manager(
        tmp_path, document_processor=process, store_factory=SerialStore
    )
    try:
        final = finish(
            manager_instance,
            manager_instance.create_files(files(tmp_path, 4), workers=2),
        )
        assert final["counts"]["succeeded"] == 4
        assert peak == 2
        assert publication_peak == 1
    finally:
        manager_instance.close(wait=True)


def test_openalex_pages_apply_query_filter_limit_and_preserve_source_snapshot(
    tmp_path,
):
    pages, snapshots, pdfs = [], [], []

    def fetch(query, cursor, per_page, mailto, filter):
        pages.append((query, cursor, per_page, filter))
        ids = [1, 2] if cursor == "*" else [2, 3, 4]
        return {
            "results": [
                {"id": f"https://openalex.org/W{i}", "title": f"Paper {i}"}
                for i in ids
            ],
            "meta": {"next_cursor": "next" if cursor == "*" else None},
        }

    def snapshot(doc, raw):
        snapshots.append(json.loads(raw))
        return doc

    def attach(doc, payload):
        pdfs.append(payload["id"])
        return doc

    instance = manager(
        tmp_path,
        source_fetcher=fetch,
        snapshot_writer=snapshot,
        fulltext_attacher=attach,
    )
    try:
        final = finish(
            instance,
            instance.create_openalex(
                "sensor technology", 3, filter="is_oa:true"
            ),
        )
        assert final["status"] == "completed"
        assert final["discovery_finished"]
        assert (
            final["counts"]["total"]
            == final["counts"]["discovered"]
            == final["counts"]["completed"]
            == 3
        )
        assert pages == [
            ("sensor technology", "*", 3, "is_oa:true"),
            ("sensor technology", "next", 1, "is_oa:true"),
        ]
        assert [payload["title"] for payload in snapshots] == [
            "Paper 1",
            "Paper 2",
            "Paper 3",
        ]
        assert len(pdfs) == 3
    finally:
        instance.close(wait=True)


def test_source_exhaustion_preserves_requested_target_without_fabricating_documents(  # noqa: E501
    tmp_path,
):
    instance = manager(
        tmp_path,
        source_fetcher=lambda *args: {
            "results": [],
            "meta": {"next_cursor": None},
        },
    )
    try:
        final = finish(
            instance,
            instance.create_openalex("rare direction", 10, fulltext=False),
        )
        assert final["status"] == "completed"
        assert final["counts"]["total"] == 10
        assert (
            final["counts"]["discovered"] == final["counts"]["completed"] == 0
        )
        assert final["discovery_finished"]
    finally:
        instance.close(wait=True)


@pytest.mark.parametrize(
    "meta",
    [None, {}, {"next_cursor": "*"}, {"next_cursor": ""}, {"next_cursor": 2}],
)
def test_openalex_invalid_cursor_fails_before_import_even_at_limit(
    tmp_path, meta
):
    page = {"results": [{"id": "https://openalex.org/W1", "title": "One"}]}
    if meta is not None:
        page["meta"] = meta
    instance = manager(tmp_path, source_fetcher=lambda *args: page)
    try:
        final = finish(
            instance, instance.create_openalex("sensors", 1, fulltext=False)
        )
        assert final["status"] == "failed"
        assert not final["documents"]
        assert not final["discovery_finished"]
        assert not instance.fixture_store.writes
    finally:
        instance.close(wait=True)


def test_openalex_job_uses_saved_api_key_without_persisting_it(
    tmp_path, monkeypatch
):
    import httpx

    from lctrend.ingest import connectors

    requests = []
    monkeypatch.setenv("OPENALEX_API_KEY", "private-openalex-key")
    monkeypatch.setenv("OPENALEX_MAILTO", "researcher@example.org")

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [{"id": "https://openalex.org/W1", "title": "One"}],
                "meta": {"next_cursor": None},
            },
        )

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(respond))
    instance = manager(tmp_path)
    try:
        final = finish(
            instance,
            instance.create_openalex(
                "sensors", 150, mode="none", fulltext=False
            ),
        )
        assert final["status"] == "completed"
        assert len(requests) == 1
        assert requests[0].headers["Authorization"] == (
            "Bearer private-openalex-key"
        )
        assert "private-openalex-key" not in str(requests[0].url)
        assert requests[0].url.params["mailto"] == "researcher@example.org"
        assert requests[0].url.params["per-page"] == "100"
        assert len(instance.fixture_store.writes) == 1
        assert "private-openalex-key" not in json.dumps(final)
        for path in instance.directory.glob("**/*.json"):
            assert "private-openalex-key" not in path.read_text(
                encoding="utf-8"
            )
    finally:
        instance.close(wait=True)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_openalex_job_returns_actionable_error_without_source_credentials(
    tmp_path, status, caplog
):
    from lctrend.ingest.connectors import SourceHTTPError

    def failed_fetch(*args):
        raise SourceHTTPError(
            status, "https://api.openalex.org/works?api_key=private-source-key"
        )

    instance = manager(tmp_path, source_fetcher=failed_fetch)
    try:
        final = finish(
            instance, instance.create_openalex("sensors", 1, fulltext=False)
        )
        assert final["status"] == "failed"
        assert final["error"]["code"] == f"http_{status}"
        assert "OpenAlex" in final["error"]["message"]
        assert "private-source-key" not in json.dumps(final)
        assert "private-source-key" not in caplog.text
    finally:
        instance.close(wait=True)


def test_mode_none_never_constructs_provider_and_publishes_metadata_only(
    tmp_path,
):
    def forbidden():
        raise AssertionError("disabled extractor was touched")

    instance = manager(tmp_path, provider_factory=forbidden)
    try:
        final = finish(
            instance, instance.create_files(files(tmp_path, 1), mode="none")
        )
        assert final["status"] == "completed"
        assert final["documents"][0]["llm_status"] == "disabled"
        assert instance.fixture_store.writes[0][1] is None
    finally:
        instance.close(wait=True)


def test_result_lookup_cannot_escape_job_directory_and_snapshots_are_detached(
    tmp_path,
):
    instance = manager(tmp_path)
    try:
        final = finish(instance, instance.create_files(files(tmp_path, 1)))
        view = instance.get_job(final["job_id"])
        view["documents"][0]["title"] = "mutated"
        assert instance.list_jobs()[0]["documents"][0]["title"] != "mutated"
        with pytest.raises(KeyError):
            instance.get_result(final["job_id"], "../../outside")
        with pytest.raises(KeyError):
            instance.get_result("../../outside", "d000001")
    finally:
        instance.close(wait=True)


@pytest.mark.parametrize(
    "workers,mode",
    [
        (0, "llm"),
        (17, "llm"),
        (True, "llm"),
        (1, "search"),
        (1, "unsupported"),
    ],
)
def test_invalid_worker_or_mode_is_rejected_before_scheduling(
    tmp_path, workers, mode
):
    instance = manager(tmp_path)
    try:
        with pytest.raises(ValueError):
            instance.create_files(
                files(tmp_path, 1), workers=workers, mode=mode
            )
        assert instance.list_jobs() == []
    finally:
        instance.close(wait=True)


def test_close_rejects_new_work(tmp_path):
    instance = manager(tmp_path)
    instance.close(wait=True)
    with pytest.raises(RuntimeError, match="closed"):
        instance.create_files(files(tmp_path, 1))


def test_source_payload_jobs_skip_discovery_and_create_provider_per_job(
    tmp_path,
):
    factories = []

    def provider_factory():
        factories.append(1)
        return object()

    def forbidden(*args):
        raise AssertionError("discovery must not be run for supplied records")

    instance = manager(
        tmp_path, source_fetcher=forbidden, provider_factory=provider_factory
    )
    try:
        one = finish(
            instance,
            instance.create_payloads(
                "openalex",
                [{"id": "https://openalex.org/W1", "title": "One"}],
                fulltext=False,
            ),
        )
        two = finish(
            instance,
            instance.create_payloads(
                "pypi",
                [
                    {
                        "info": {"name": "sensor-kit", "version": "1.0"},
                        "releases": {},
                        "urls": [],
                    }
                ],
            ),
        )
        assert one["status"] == two["status"] == "completed"
        assert factories == [1, 1]
        assert len(instance.fixture_store.writes) == 2
    finally:
        instance.close(wait=True)


def test_nonterminal_llm_stage_success_does_not_mark_branch_completed(
    tmp_path,
):
    started, release = Event(), Event()

    def process(doc, event, **kwargs):
        event({"branch": "llm", "stage": "extract", "status": "running"})
        event({"branch": "llm", "stage": "extract", "status": "succeeded"})
        started.set()
        assert release.wait(3)
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    try:
        job = instance.create_files(files(tmp_path, 1))
        assert started.wait(3)
        assert (
            instance.get_job(job["job_id"])["documents"][0]["llm_status"]
            == "running"
        )
        release.set()
        finish(instance, job)
    finally:
        release.set()
        instance.close(wait=True)


def test_atomic_json_retries_transient_windows_file_lock(
    tmp_path, monkeypatch
):
    import os

    from frontend.server.jobs import _write_json

    real_replace = os.replace
    attempts = []

    def replace(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("brief sharing violation")
        return real_replace(source, destination)

    monkeypatch.setattr("frontend.server.jobs.os.replace", replace)
    path = tmp_path / "job.json"
    _write_json(path, {"complete": True})
    assert json.loads(path.read_text()) == {"complete": True}
    assert len(attempts) == 3


def test_atomic_json_never_swallows_persistent_permission_failure(
    tmp_path, monkeypatch
):
    from frontend.server.jobs import _write_json

    def forbidden(*args):
        raise PermissionError("persistent permission failure")

    monkeypatch.setattr("frontend.server.jobs.os.replace", forbidden)
    with pytest.raises(PermissionError):
        _write_json(tmp_path / "job.json", {"complete": True})
    assert list(tmp_path.glob(".job-*.tmp")) == []


def test_child_registration_is_durable_before_any_model_or_graph_call(
    tmp_path,
):
    registered = []

    def process(doc, **kwargs):
        assert registered
        assert registered[0]["documents"][0]["doc_id"] == "d000001"
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    try:

        def register(job):
            persisted = json.loads(
                (instance.directory / job["job_id"] / "job.json").read_text(
                    encoding="utf-8"
                )
            )
            assert persisted["status"] == "queued"
            assert instance.fixture_store.events == []
            registered.append(job)

        final = finish(
            instance,
            instance.create_payloads(
                "openalex",
                [{"id": "https://openalex.org/W1"}],
                fulltext=False,
                on_created=register,
            ),
        )
        assert final["status"] == "completed"
    finally:
        instance.close(wait=True)


def test_registration_failure_never_dispatches_a_child_job(tmp_path):
    instance = manager(tmp_path)
    try:

        def rejected(job):
            raise RuntimeError("ledger write failed")

        with pytest.raises(RuntimeError):
            instance.create_payloads(
                "openalex",
                [{"id": "https://openalex.org/W1"}],
                on_created=rejected,
            )
        assert instance._futures == {}
        assert instance.fixture_store.events == []
        assert instance.list_jobs()[0]["status"] == "failed"
    finally:
        instance.close(wait=True)


def test_republication_skips_provider_pdf_and_model_calls(
    tmp_path,
):
    processed, writes = [], []

    class RetryStore(Store):
        def write_processed(self, doc, result):
            writes.append(result.run.run_id)
            if len(writes) == 1:
                raise RuntimeError("temporary database failure")

    def process(doc, **kwargs):
        processed.append(doc.document_id)
        return extraction(doc)

    instance = manager(
        tmp_path, store_factory=RetryStore, document_processor=process
    )
    payloads = [{"id": "https://openalex.org/W1", "title": "One"}]
    try:
        first = finish(
            instance,
            instance.create_payloads("openalex", payloads, fulltext=False),
        )
        old_doc = first["documents"][0]
        assert old_doc["stage"] == "publication" and old_doc["result_ready"]
        cached = instance.get_result(first["job_id"], old_doc["doc_id"])

        def forbidden(*args):
            raise AssertionError(
                "republication must not touch any model or source"
            )

        instance._provider_factory = forbidden
        instance._pdf_support_checker = instance._snapshot_writer = forbidden
        instance._document_processor = forbidden
        second = finish(
            instance,
            instance.create_payloads(
                "openalex", payloads, cached_results=[cached]
            ),
        )
        assert second["status"] == "completed"
        assert len(processed) == 1 and len(writes) == 2
        assert writes[0] == writes[1]
    finally:
        instance.close(wait=True)


def test_failed_or_invalid_cached_extraction_cannot_be_republished(tmp_path):
    doc = document(files(tmp_path, 1)[0])
    instance = manager(tmp_path)
    try:
        cached = {
            "document": doc.model_dump(mode="json"),
            "extraction": extraction(doc, "failed").model_dump(mode="json"),
        }
        with pytest.raises(ValueError):
            instance.create_payloads(
                "openalex",
                [{"id": "https://openalex.org/W1"}],
                cached_results=[cached],
            )
        cached["extraction"]["run"]["status"] = "succeeded"
        cached["extraction"]["document_version_id"] = "another-version"
        with pytest.raises(ValueError):
            instance.create_payloads(
                "openalex",
                [{"id": "https://openalex.org/W1"}],
                cached_results=[cached],
            )
        assert instance.list_jobs() == []
    finally:
        instance.close(wait=True)


def test_terminal_cache_is_bounded_and_old_results_remain_readable(tmp_path):
    instance = manager(tmp_path, max_cached_jobs=3)
    first = None
    try:
        for index in range(8):
            job = finish(
                instance,
                instance.create_payloads(
                    "openalex",
                    [{"id": f"https://openalex.org/W{index}"}],
                    fulltext=False,
                ),
            )
            first = first or job
        assert len(instance._jobs) <= 3
        assert len(instance._tasks) <= 3
        assert len(instance._futures) <= 3
        assert len(instance._cancel) <= 3
        assert first["job_id"] not in instance._jobs
        assert instance.get_job(first["job_id"])["status"] == "completed"
        assert instance.get_result(first["job_id"], "d000001")["extraction"]
        assert instance.cancel_job(first["job_id"])["status"] == "completed"
    finally:
        instance.close(wait=True)


def test_restart_preserves_interrupted_publication_stage(tmp_path):
    instance = manager(tmp_path)
    final = finish(
        instance,
        instance.create_payloads(
            "openalex",
            [{"id": "https://openalex.org/W1"}],
            fulltext=False,
        ),
    )
    instance.close(wait=True)
    final["status"] = "running"
    final["documents"][0].update(status="running", stage="publication")
    path = instance.directory / final["job_id"] / "job.json"
    path.write_text(json.dumps(final), encoding="utf-8")
    restored = JobManager(instance.directory)
    try:
        doc = restored.get_job(final["job_id"])["documents"][0]
        assert doc["stage"] == "interrupted"
        assert doc["interrupted_stage"] == "publication"
        assert doc["result_ready"]
        assert (
            restored.get_result(final["job_id"], doc["doc_id"])["extraction"][
                "run"
            ]["status"]
            == "succeeded"
        )
    finally:
        restored.close(wait=True)


def test_failed_page_lets_running_documents_finish(tmp_path):
    # G-2: a 429 on the next page must not abandon paid extractions.
    started = Barrier(3, timeout=5)

    def fetch(query, cursor, per_page, mailto, filter):
        if cursor == "*":
            return {
                "results": [
                    {"id": f"https://openalex.org/W{i}", "title": f"P{i}"}
                    for i in (1, 2)
                ],
                "meta": {"next_cursor": "next"},
            }
        started.wait()  # both documents are being processed
        raise RuntimeError("HTTP 429")

    def process(doc, **kwargs):
        started.wait()
        sleep(0.1)
        return extraction(doc)

    instance = manager(
        tmp_path,
        source_fetcher=fetch,
        document_processor=process,
        fulltext_attacher=lambda doc, payload: doc,
    )
    try:
        final = finish(
            instance, instance.create_openalex("sensors", 5, workers=2)
        )
        assert final["status"] == "failed"
        assert [doc["status"] for doc in final["documents"]] == [
            "succeeded",
            "succeeded",
        ]
        assert len(instance.fixture_store.writes) == 2
    finally:
        instance.close(wait=True)


def test_failing_error_handler_fails_one_document_only(tmp_path):
    # G-3: an error inside the error handler must not end the job while
    # other documents run (the crawl would then run them twice).
    calls = []

    def process(doc, **kwargs):
        calls.append(doc.document_id)
        if doc.document_id == "study-0":
            raise RuntimeError("model failed")
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    original = instance._update_document

    def update(job_id, doc_id, force=None, **updates):
        if updates.get("status") == "failed":
            raise OSError("disk full")
        return original(job_id, doc_id, force=force, **updates)

    instance._update_document = update
    try:
        final = finish(
            instance, instance.create_files(files(tmp_path, 2), workers=2)
        )
        statuses = {doc["title"]: doc["status"] for doc in final["documents"]}
        assert statuses == {
            "study-0.txt": "failed",
            "study-1.txt": "succeeded",
        }
        assert final["status"] == "completed"
        assert sorted(calls) == ["study-0", "study-1"]
    finally:
        instance.close(wait=True)


def test_finished_job_has_only_terminal_documents(tmp_path):
    instance = manager(tmp_path)
    try:
        job = {
            "job_id": "a" * 32,
            "limit": 3,
            "status": "running",
            "created_at": "2026-09-28T00:00:00+00:00",
            "documents": [
                {
                    "doc_id": "d1",
                    "status": "running",
                    "stage": "publication",
                    "llm_status": "running",
                },
                {
                    "doc_id": "d2",
                    "status": "queued",
                    "stage": "queued",
                    "llm_status": "queued",
                },
                {
                    "doc_id": "d3",
                    "status": "succeeded",
                    "stage": "done",
                    "llm_status": "succeeded",
                },
            ],
        }
        with instance._lock:
            instance._jobs[job["job_id"]] = job
            instance._finish(job, "interrupted")
        assert [doc["status"] for doc in job["documents"]] == [
            "failed",
            "cancelled",
            "succeeded",
        ]
        assert job["documents"][0]["interrupted_stage"] == "publication"
        assert job["documents"][0]["llm_status"] == "failed"
        assert job["counts"]["running"] == 0
    finally:
        instance.close(wait=True)


def test_skipped_no_text_run_is_reported_as_no_text(tmp_path):
    # A-10: the pipeline's own status now says why nothing was extracted.
    instance = manager(
        tmp_path,
        file_parser=lambda path: document(path).model_copy(
            update={"chunks": [], "coverage": "metadata_only"}
        ),
        document_processor=lambda doc, **kwargs: extraction(
            doc, "skipped_no_text", coverage={"total_chunks": 0}
        ),
    )
    try:
        final = finish(instance, instance.create_files(files(tmp_path, 1)))
        record = final["documents"][0]
        assert record["llm_status"] == "no_text"
        assert record["error"]["code"] == "no_text"
    finally:
        instance.close(wait=True)


def test_stop_lets_a_running_document_finish_within_the_grace(tmp_path):
    # G-4: the server stopped at once and the running document's (paid)
    # result was lost.
    started = Event()

    def process(doc, **kwargs):
        started.set()
        sleep(0.2)
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    job = instance.create_files(files(tmp_path, 1))
    assert started.wait(5)
    instance.close(timeout=5)
    final = json.loads(
        (tmp_path / "jobs" / job["job_id"] / "job.json").read_text("utf-8")
    )
    assert final["documents"][0]["status"] == "succeeded"
    assert len(instance.fixture_store.writes) == 1


def test_stop_past_the_grace_records_the_job_as_interrupted(tmp_path):
    started, release = Event(), Event()

    def process(doc, **kwargs):
        started.set()
        release.wait(10)
        return extraction(doc)

    instance = manager(tmp_path, document_processor=process)
    job = instance.create_files(files(tmp_path, 1))
    try:
        assert started.wait(5)
        instance.close(timeout=0.2)
        final = json.loads(
            (tmp_path / "jobs" / job["job_id"] / "job.json").read_text(
                "utf-8"
            )
        )
        assert final["status"] == "interrupted"
        document = final["documents"][0]
        assert document["status"] == "failed"
        assert document["error"]["code"] == "interrupted"
    finally:
        release.set()


def test_graph_connection_drop_is_retried(monkeypatch):
    import asyncio

    from neo4j.exceptions import ServiceUnavailable

    from frontend.server import jobs

    pauses, calls = [], []

    async def no_sleep(seconds):
        pauses.append(seconds)

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ServiceUnavailable("connection reset")
        return "ok"

    monkeypatch.setattr(jobs.asyncio, "sleep", no_sleep)
    assert asyncio.run(jobs._graph_retry(flaky, "job")) == "ok"
    assert pauses == [2.0, 5.0]

    def broken():
        raise ValueError("bad query")

    try:
        asyncio.run(jobs._graph_retry(broken, "job"))
    except ValueError:
        pass
    assert pauses == [2.0, 5.0], "a non-connection error is not retried"
