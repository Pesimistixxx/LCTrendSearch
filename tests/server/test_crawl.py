"""Queue and checkpoint tests without network, Neo4j or model requests."""

from copy import deepcopy
from threading import Event, RLock
from time import monotonic, sleep
from uuid import uuid4

import pytest

from frontend.server.crawl import CrawlManager


def item(source, name, *, canonical=None, payload=None):
    return {
        "source": source,
        "source_id": name,
        "canonical_id": canonical or f"{source}:{name}",
        "title": name,
        "url": f"https://example.org/{name}",
        "payload": payload if payload is not None else {"name": name},
    }


def page(items=(), cursor=None, *, complete=True, total=None, limitations=()):
    return {
        "items": list(items),
        "next_cursor": cursor,
        "complete": complete,
        "total": len(items) if total is None else total,
        "limitations": list(limitations),
    }


class Jobs:
    def __init__(self, directory, *, blocked=False, statuses=None):
        self.directory = directory
        self.jobs = {}
        self.calls = []
        self.started, self.release = Event(), Event()
        self.blocked = blocked
        self.statuses = statuses or {}
        self._lock = RLock()

    def create_payloads(self, source, payloads, **kwargs):
        with self._lock:
            self.calls.append((source, deepcopy(payloads)))
            job_id = uuid4().hex
            docs = [
                {
                    "doc_id": f"d{index:06d}",
                    "status": "running"
                    if index == 1 and self.blocked
                    else "queued",
                    "error": None,
                    "result_ready": False,
                }
                for index in range(1, len(payloads) + 1)
            ]
            job = {
                "job_id": job_id,
                "status": "running" if self.blocked else "completed",
                "documents": docs,
                "error": None,
            }
            if not self.blocked:
                for doc, payload in zip(docs, payloads):
                    doc["status"] = self.statuses.get(
                        payload.get("name"), "succeeded"
                    )
                    doc["result_ready"] = True
            self.jobs[job_id] = job
            if kwargs.get("on_created") is not None:
                kwargs["on_created"](deepcopy(job))
            self.started.set()
            return deepcopy(job)

    def get_job(self, job_id):
        with self._lock:
            job = self.jobs[job_id]
            if (
                job["status"] in {"running", "cancelling"}
                and self.release.is_set()
            ):
                was_cancelled = job["status"] == "cancelling"
                job["status"] = "cancelled" if was_cancelled else "completed"
                for index, doc in enumerate(job["documents"]):
                    doc["status"] = (
                        "succeeded"
                        if index == 0 or not was_cancelled
                        else "cancelled"
                    )
                    doc["result_ready"] = doc["status"] == "succeeded"
            return deepcopy(job)

    def cancel_job(self, job_id):
        with self._lock:
            self.jobs[job_id]["status"] = "cancelling"
            return deepcopy(self.jobs[job_id])


def manager(tmp_path, jobs=None, **overrides):
    jobs = jobs or Jobs(tmp_path / "jobs")
    settings = {
        "discoverers": {"openalex": lambda *args: page()},
        "hydrator": lambda item: item["payload"],
        "pypi_discoverer": lambda payload: [],
        "domains": [
            {"name": "Domain A", "aliases": ["domain a", "Alias A"]},
            {"name": "Domain B", "aliases": []},
        ],
        # No graph: the ledger alone decides (graph tests pass a reader).
        "processed_reader": lambda: None,
    }
    settings.update(overrides)
    return CrawlManager(jobs, tmp_path / "ledger", **settings)


def finish(instance, crawl):
    instance._futures[crawl["crawl_id"]].result(timeout=5)
    return instance.get_crawl(crawl["crawl_id"])


def until(check):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        if check():
            return
        sleep(0.02)
    raise AssertionError("Expected progress update did not arrive")


def test_progress_updates_after_each_document_before_child_batch_finishes(
    tmp_path,
):
    jobs = Jobs(tmp_path / "jobs", blocked=True)
    instance = manager(
        tmp_path,
        jobs,
        discoverers={
            "openalex": lambda *args: page(
                [item("openalex", "a"), item("openalex", "b")]
            )
        },
    )
    try:
        crawl = instance.create("Sensors")
        assert jobs.started.wait(3)
        until(
            lambda: (
                instance.get_crawl(crawl["crawl_id"])["counts"]["processing"]
                == 1
            )
        )
        assert instance.get_crawl(crawl["crawl_id"])["counts"]["pending"] == 1
        with jobs._lock:
            job = next(iter(jobs.jobs.values()))
            job["documents"][0].update(
                status="succeeded", stage="done", result_ready=True
            )
            job["documents"][1].update(
                status="running",
                stage="review",
                llm_status="running",
            )
        until(
            lambda: (
                instance.get_crawl(crawl["crawl_id"])["counts"]["parsed"] == 1
            )
        )
        state = instance.get_crawl(crawl["crawl_id"])
        assert state["status"] == "running" and state["stage"] == "review"
        assert (
            state["counts"]["processing"] == 1
            and state["counts"]["pending"] == 0
        )
        active = instance.list_materials(
            crawl["crawl_id"], status="processing"
        )["items"][0]
        assert (
            active["stage"] == "review" and active["llm_status"] == "running"
        )
        jobs.release.set()
        assert finish(instance, crawl)["counts"]["parsed"] == 2
    finally:
        jobs.release.set()
        instance.close(wait=True)


def test_graph_seed_generator_is_consumed_before_driver_is_closed(tmp_path):
    jobs = Jobs(tmp_path / "jobs")

    class Store:
        closed = False

        def verify_connectivity(self):
            assert not self.closed

        def processed_materials(self):
            assert not self.closed
            yield {
                "source": "openalex",
                "source_id": "known",
                "canonical_id": "openalex:known",
                "status": "parsed",
            }
            assert not self.closed

        def close(self):
            self.closed = True

    # A driver per call, as GraphStore: the graph identity is read first.
    stores = []
    jobs._store_factory = lambda: stores.append(Store()) or stores[-1]
    instance = manager(
        tmp_path,
        jobs,
        processed_reader=None,
        discoverers={
            "openalex": lambda *args: page([item("openalex", "known")])
        },
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        assert final["counts"]["parsed"] == 1 and jobs.calls == []
        assert stores and all(store.closed for store in stores)
    finally:
        instance.close(wait=True)


def test_global_dedup_across_pages_aliases_domains_topics_and_restart(
    tmp_path,
):
    queries = []

    def discover(topic, cursor):
        queries.append((topic, cursor))
        return page([item("openalex", topic, canonical="doi:10.1/shared")])

    instance = manager(tmp_path, discoverers={"openalex": discover})
    first = finish(instance, instance.create())
    assert {query for query, _ in queries} == {
        "Domain A",
        "Alias A",
        "Domain B",
    }
    assert first["status"] == "completed"
    assert first["counts"]["discovered"] == first["counts"]["parsed"] == 1
    assert (
        first["counts"]["duplicates"] == 1
    )  # Another domain, not a repeated same-domain alias.
    assert len(instance.job_manager.calls) == 1
    instance.close(wait=True)

    restored = manager(tmp_path, discoverers={"openalex": discover})
    try:
        second = finish(restored, restored.create("Another topic"))
        assert (
            second["counts"]["parsed"] == second["counts"]["duplicates"] == 1
        )
        assert restored.job_manager.calls == []
        records = restored.list_materials(second["crawl_id"])
        assert records["total"] == 1
        assert records["items"][0]["job_id"]
    finally:
        restored.close(wait=True)


def test_graph_decides_duplicates_not_the_ledger(tmp_path):
    graph = []

    def held(name):
        return {
            "source": "openalex",
            "source_id": name,
            "canonical_id": f"doi:10.1/{name}",
            "status": "parsed",
        }

    def discover(topic, cursor):
        return page(
            [
                item("openalex", "lost", canonical="doi:10.1/lost"),
                item("openalex", "kept", canonical="doi:10.1/kept"),
            ]
        )

    instance = manager(
        tmp_path,
        discoverers={"openalex": discover},
        processed_reader=lambda: list(graph),
    )
    try:
        first = finish(instance, instance.create("Sensors"))
        assert first["counts"]["parsed"] == 2
        assert len(instance.job_manager.calls) == 1
        graph.extend([held("lost"), held("kept")])  # published

        # Nodes deleted in the same database: one of the two is gone.
        graph.remove(held("lost"))
        second = finish(instance, instance.create("Another topic"))
        assert second["counts"]["parsed"] == 2
        assert [
            [payload["name"] for payload in payloads]
            for _, payloads in instance.job_manager.calls[1:]
        ] == [["lost"]]
        graph.append(held("lost"))

        # Everything is in the graph again: nothing is repeated.
        finish(instance, instance.create("Third topic"))
        assert len(instance.job_manager.calls) == 2
    finally:
        instance.close(wait=True)


def test_all_pages_follow_cursors_without_a_document_limit(tmp_path):
    visited = []

    def discover(topic, cursor):
        visited.append(cursor)
        index = 1 if cursor == "*" else int(cursor)
        return page(
            [item("openalex", str(index))],
            str(index + 1) if index < 4 else None,
            complete=index == 4,
            total=4,
        )

    instance = manager(
        tmp_path, discoverers={"openalex": discover}, batch_size=1
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        assert visited == ["*", "2", "3", "4"]
        assert final["counts"]["parsed"] == 4
        source = next(
            source
            for source in final["sources"]
            if source["source"] == "openalex"
        )
        assert source["complete"] and source["total"] == 4
    finally:
        instance.close(wait=True)


def test_limit_caps_each_direction_and_source_across_aliases(tmp_path):
    visited = []

    def discover(topic, cursor):
        visited.append((topic, cursor))
        index = 1 if cursor == "*" else int(cursor)
        return page(
            [item("openalex", f"{topic}-{index}-{n}") for n in range(3)],
            str(index + 1),
            complete=False,
            total=1000,
        )

    instance = manager(
        tmp_path, discoverers={"openalex": discover}, batch_size=100
    )
    try:
        final = finish(instance, instance.create(limit=4))
        assert final["status"] == "completed" and final["limit"] == 4
        # Domain A has two queries, yet shares one budget of four.
        assert visited == [
            ("Alias A", "*"),
            ("Domain A", "*"),
            ("Domain B", "*"),
            ("Domain B", "2"),
        ]
        by_name = {
            direction["name"]: direction["sources"]["openalex"]
            for direction in final["directions"]
        }
        assert by_name["Domain A"]["discovered"] == 4
        assert by_name["Domain B"]["discovered"] == 4
        assert by_name["Domain A"]["parsed"] == 4
        source = next(
            source
            for source in final["sources"]
            if source["source"] == "openalex"
        )
        assert source["status"] == "capped" and not source["complete"]
        assert any(
            item["code"] == "user_limit" for item in source["limitations"]
        )
    finally:
        instance.close(wait=True)


def test_invalid_limit_is_rejected(tmp_path):
    instance = manager(tmp_path)
    try:
        for value in [0, 10001, True, "5"]:
            with pytest.raises(ValueError):
                instance.create(limit=value)
    finally:
        instance.close(wait=True)


def test_default_directions_follow_catalog_selection(tmp_path, monkeypatch):
    import frontend.server.crawl as crawl

    catalog = {
        "domains": [
            {"name": "Kept", "aliases": []},
            {"name": "Skipped", "aliases": []},
        ],
        "crawl_directions": ["Kept"],
        "platforms": {},
    }
    monkeypatch.setattr(crawl, "load_catalog", lambda name: catalog)
    instance = manager(tmp_path, domains=None)
    try:
        assert [domain["name"] for domain in instance._domains] == ["Kept"]
    finally:
        instance.close(wait=True)


def test_source_failure_does_not_stop_other_sources_or_discovered_materials(
    tmp_path,
):
    calls = []

    def failed(*args):
        calls.append("failed")
        raise RuntimeError("secret key and private URL")

    instance = manager(
        tmp_path,
        discoverers={
            "openalex": failed,
            "github": lambda *args: page([item("github", "repo")]),
        },
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        assert final["status"] == "failed"
        assert final["counts"]["parsed"] == 1
        assert calls == ["failed"]
        source = next(
            source
            for source in final["sources"]
            if source["source"] == "openalex"
        )
        assert source["status"] == "failed" and not source["complete"]
        assert "secret" not in str(final)
        assert final["error"]["code"] == "source_failures"
    finally:
        instance.close(wait=True)


def test_limited_github_and_link_only_pypi_never_claim_full_coverage(tmp_path):
    limited = [
        {
            "code": "github_search_cap",
            "message": "Search covers only the first 1000",
            "reported_total": 5000,
        }
    ]
    instance = manager(
        tmp_path,
        discoverers={
            "github": lambda *args: page(
                [item("github", "repo")],
                complete=False,
                total=1000,
                limitations=limited,
            )
        },
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        github = next(
            source
            for source in final["sources"]
            if source["source"] == "github"
        )
        pypi = next(
            source for source in final["sources"] if source["source"] == "pypi"
        )
        epo = next(
            source for source in final["sources"] if source["source"] == "epo"
        )
        assert (
            github["status"] == "limited"
            and github["total"] == 1000
            and not github["complete"]
        )
        assert github["limitations"] == limited
        assert pypi["status"] == "linked" and not pypi["complete"]
        assert epo["status"] == "unsupported" and not epo["complete"]
    finally:
        instance.close(wait=True)


def test_github_hydration_finds_pypi_before_llm_and_reuses_metadata(
    tmp_path,
):
    hydrated = []
    jobs = Jobs(tmp_path / "jobs", statuses={"repo": "failed"})

    def hydrate(record):
        hydrated.append(record["source"])
        return {"name": record["source_id"], "readme": "link"}

    instance = manager(
        tmp_path,
        jobs,
        discoverers={"github": lambda *args: page([item("github", "repo")])},
        hydrator=hydrate,
        pypi_discoverer=lambda payload: [
            item("pypi", "pkg", payload={"name": "pkg"})
        ],
    )
    try:
        first = finish(instance, instance.create("First topic"))
        assert first["counts"]["failed"] == first["counts"]["parsed"] == 1
        second = finish(instance, instance.create("Second topic"))
        assert second["counts"]["discovered"] == 2
        assert second["counts"]["duplicates"] == 2
        assert hydrated == ["github", "pypi"]
        assert [source for source, _ in jobs.calls] == ["github", "pypi"]
    finally:
        instance.close(wait=True)


def test_pause_keeps_pending_and_resumes_saved_cursor(
    tmp_path,
):
    jobs = Jobs(tmp_path / "jobs", blocked=True)
    visits = []

    def discovery(topic, cursor):
        visits.append(cursor)
        return (
            page(
                [item("openalex", "a"), item("openalex", "b")],
                "next",
                complete=False,
            )
            if cursor == "*"
            else page([item("openalex", "c")])
        )

    instance = manager(tmp_path, jobs, discoverers={"openalex": discovery})
    try:
        crawl = instance.create("Sensors")
        assert jobs.started.wait(3)
        assert instance.pause(crawl["crawl_id"])["status"] == "pausing"
        jobs.release.set()
        paused = finish(instance, crawl)
        assert paused["status"] == "paused"
        assert paused["counts"]["parsed"] == paused["counts"]["pending"] == 1
        assert visits == ["*"]
        jobs.blocked = False
        final = finish(instance, instance.resume(crawl["crawl_id"]))
        assert visits == ["*", "next"]
        assert final["counts"]["parsed"] == 3
        names = [
            payload["name"]
            for _, payloads in jobs.calls
            for payload in payloads
        ]
        assert names.count("a") == 1 and names.count("b") == 2
        # b was queued in the cancelled first job and processed only on resume.
    finally:
        jobs.release.set()
        instance.close(wait=True)


def test_restart_pauses_without_processing_or_paid_repeat(
    tmp_path,
):
    instance = manager(tmp_path)
    first = finish(instance, instance.create("Sensors"))
    with instance._lock, instance._db:
        instance._db.execute(
            "UPDATE crawls SET status='running' WHERE crawl_id=?",
            (first["crawl_id"],),
        )
    instance.close(wait=True)
    restored = manager(tmp_path)
    try:
        assert restored.get_crawl(first["crawl_id"])["status"] == "paused"
        assert restored._futures == {}
        assert restored.job_manager.calls == []
    finally:
        restored.close(wait=True)


def test_partial_and_failed_are_not_automatically_reprocessed(tmp_path):
    jobs = Jobs(
        tmp_path / "jobs", statuses={"partial": "partial", "failed": "failed"}
    )

    def discovery(*args):
        return page([item("openalex", "partial"), item("openalex", "failed")])

    instance = manager(tmp_path, jobs, discoverers={"openalex": discovery})
    try:
        first = finish(instance, instance.create("Sensors"))
        assert first["counts"]["partial"] == first["counts"]["failed"] == 1
        final = finish(instance, instance.resume(first["crawl_id"]))
        assert final["counts"]["partial"] == final["counts"]["failed"] == 1
        assert len(jobs.calls) == 1
        jobs.statuses["failed"] = "succeeded"
        retried = finish(instance, instance.retry_failed(first["crawl_id"]))
        assert retried["counts"]["parsed"] == retried["counts"]["partial"] == 1
        assert [payload["name"] for payload in jobs.calls[-1][1]] == ["failed"]
    finally:
        instance.close(wait=True)


def test_neo4j_seed_prevents_reprocessing_previous_cli_results(tmp_path):
    seed_calls = []

    def seed():
        seed_calls.append(1)
        return [
            {
                "source": "openalex",
                "source_id": "old",
                "canonical_id": "doi:10.1/old",
                "title": "Previously parsed",
                "status": "parsed",
                "url": "https://example.org/old",
            }
        ]

    instance = manager(
        tmp_path,
        processed_reader=seed,
        discoverers={
            "openalex": lambda *args: page(
                [item("openalex", "new-id", canonical="doi:10.1/old")]
            )
        },
    )
    try:
        first = finish(instance, instance.create("Sensors"))
        second = finish(instance, instance.create("Another topic"))
        # Read on every run: the graph can change while the server runs.
        assert seed_calls == [1, 1]
        assert first["counts"]["parsed"] == second["counts"]["parsed"] == 1
        assert instance.job_manager.calls == []
        record = instance.list_materials(first["crawl_id"])["items"][0]
        assert record["job_id"] is None and record["doc_id"] is None
    finally:
        instance.close(wait=True)


def test_material_pagination_is_bounded_filterable_and_uses_global_unique_rows(
    tmp_path,
):
    jobs = Jobs(tmp_path / "jobs", statuses={"1": "failed", "2": "partial"})
    instance = manager(
        tmp_path,
        jobs,
        discoverers={
            "openalex": lambda *args: page(
                [item("openalex", str(i)) for i in range(120)]
            )
        },
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        first = instance.list_materials(final["crawl_id"], limit=100)
        second = instance.list_materials(final["crawl_id"], offset=100)
        assert first["total"] == second["total"] == 120
        assert len(first["items"]) == 100 and len(second["items"]) == 20
        assert not {row["material_id"] for row in first["items"]} & {
            row["material_id"] for row in second["items"]
        }
        failed = instance.list_materials(final["crawl_id"], status="failed")
        assert failed["total"] == 1
        with pytest.raises(ValueError):
            instance.list_materials(final["crawl_id"], limit=101)
        with pytest.raises(KeyError):
            instance.list_materials("unknown")
    finally:
        instance.close(wait=True)


def test_source_failure_retried_only_by_explicit_resume(tmp_path):
    calls = []

    def discovery(*args):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("rate limit")
        return page([item("openalex", "new")])

    instance = manager(tmp_path, discoverers={"openalex": discovery})
    try:
        first = finish(instance, instance.create("Sensors"))
        assert first["status"] == "failed" and len(calls) == 1
        final = finish(instance, instance.resume(first["crawl_id"]))
        assert final["status"] == "completed" and len(calls) == 2
        assert final["counts"]["parsed"] == 1
    finally:
        instance.close(wait=True)


def test_early_coverage_warning_survives_later_complete_page(tmp_path):
    warning = {
        "code": "github_incomplete_results",
        "message": "First page was incomplete",
    }

    def discovery(topic, cursor):
        return (
            page(
                [item("github", "one")],
                "2",
                complete=False,
                limitations=[warning],
            )
            if cursor == "*"
            else page([item("github", "two")], complete=True)
        )

    instance = manager(tmp_path, discoverers={"github": discovery})
    try:
        final = finish(instance, instance.create("Sensors"))
        source = next(
            source
            for source in final["sources"]
            if source["source"] == "github"
        )
        assert source["status"] == "limited" and not source["complete"]
        assert source["limitations"] == [warning]
    finally:
        instance.close(wait=True)


def test_graph_seeded_github_loads_package_links_without_repeating_llm(
    tmp_path,
):
    seed = [
        {
            "source": "github",
            "source_id": "owner/repo",
            "canonical_id": "github:owner/repo",
            "status": "parsed",
        }
    ]
    hydrated = []

    def hydrate(record):
        hydrated.append(record["source"])
        return {"name": record["source_id"], "readme": "package-link"}

    instance = manager(
        tmp_path,
        processed_reader=lambda: seed,
        hydrator=hydrate,
        discoverers={
            "github": lambda *args: page([item("github", "owner/repo")])
        },
        pypi_discoverer=lambda payload: [item("pypi", "linked-package")],
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        assert final["counts"]["parsed"] == 2
        assert hydrated == ["github", "pypi"]
        assert [source for source, _ in instance.job_manager.calls] == ["pypi"]
    finally:
        instance.close(wait=True)


def test_only_interrupted_publication_is_eligible_for_cached_result(tmp_path):
    jobs = Jobs(tmp_path / "jobs")
    job = {
        "job_id": "old",
        "status": "interrupted",
        "documents": [
            {
                "doc_id": "d000001",
                "status": "failed",
                "stage": "interrupted",
                "interrupted_stage": "publication",
                "result_ready": True,
            }
        ],
    }
    jobs.jobs["old"] = job
    cached = {"extraction": {"run": {"status": "succeeded"}}}
    jobs.get_result = lambda *args: cached
    instance = manager(tmp_path, jobs)
    try:
        record = {"job_id": "old", "doc_id": "d000001", "stage": "review"}
        assert instance._cached_publication(record) == cached
        job["documents"][0]["interrupted_stage"] = "processing"
        assert instance._cached_publication(record) is None
        job["documents"][0]["interrupted_stage"] = "publication"
        cached["extraction"]["run"]["status"] = "failed"
        assert instance._cached_publication(record) is None
    finally:
        instance.close(wait=True)


def test_failed_seed_metadata_does_not_break_a_later_duplicate_page(tmp_path):
    visits = []
    available = False

    def discover(topic, cursor):
        visits.append(cursor)
        records = [item("github", "known")]
        if cursor != "*":
            records.append(item("github", "new"))
        return page(records, "2" if cursor == "*" else None)

    def hydrate(record):
        if record["source_id"] == "known" and not available:
            raise RuntimeError("metadata unavailable")
        return record["payload"] or {"name": record["source_id"]}

    instance = manager(
        tmp_path,
        discoverers={"github": discover},
        hydrator=hydrate,
        pypi_discoverer=lambda payload: (
            [item("pypi", "linked-package")]
            if payload.get("name") == "known"
            else []
        ),
        # The graph holds the seed and whatever the jobs published.
        processed_reader=lambda: [
            {
                "source": source,
                "source_id": name,
                "canonical_id": f"{source}:{name}",
                "status": "parsed",
            }
            for source, name in [
                ("github", "known"),
                *(
                    (source, payload["name"])
                    for source, payloads in instance.job_manager.calls
                    for payload in payloads
                ),
            ]
        ],
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        assert visits == ["*", "2"]
        assert final["status"] == "failed"
        assert final["counts"]["parsed"] == 2
        source = next(
            row for row in final["sources"] if row["source"] == "github"
        )
        assert source["status"] == "failed"
        assert not source["complete"]
        assert source["error"]
        available = True
        resumed = finish(instance, instance.resume(final["crawl_id"]))
        assert resumed["status"] == "completed"
        assert resumed["counts"]["parsed"] == 3
        assert visits == ["*", "2"]
        processed = [
            payload["name"]
            for _, payloads in instance.job_manager.calls
            for payload in payloads
        ]
        assert processed == ["new", "linked-package"]
    finally:
        instance.close(wait=True)


def test_batch_is_hydrated_concurrently_and_one_failure_stays_local(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("frontend.server.crawl.default_workers", lambda: 4)
    lock, running, peak = RLock(), [0], [0]

    def hydrate(record):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        sleep(0.05)
        with lock:
            running[0] -= 1
        if record["source_id"] == "broken":
            raise RuntimeError("card unavailable")
        return {"name": record["source_id"], "description": "text"}

    names = ["a", "b", "broken", "c"]
    instance = manager(
        tmp_path,
        hydrator=hydrate,
        discoverers={
            "hh": lambda *args: page(
                [item("hh", name, payload={}) for name in names]
            )
        },
    )
    try:
        final = finish(instance, instance.create("Sensors"))
        assert peak[0] > 1
        assert final["counts"]["parsed"] == 3
        assert final["counts"]["failed"] == 1
        [(source, payloads)] = instance.job_manager.calls
        assert source == "hh"
        assert [payload["name"] for payload in payloads] == ["a", "b", "c"]
    finally:
        instance.close(wait=True)


def test_next_batch_starts_while_a_slow_batch_drains(tmp_path, monkeypatch):
    """One unfinished document must not idle the other workers: with two
    jobs allowed, the next batch is submitted before the first finishes."""
    monkeypatch.setenv("LCTREND_WORKERS", "4")

    class OverlappingJobs(Jobs):
        max_active_jobs = 2

    jobs = OverlappingJobs(tmp_path / "jobs", blocked=True)
    names = ["a", "b", "c", "d"]
    instance = manager(
        tmp_path,
        jobs,
        batch_size=2,
        discoverers={
            "openalex": lambda topic, cursor: page(
                [item("openalex", name) for name in names]
            )
        },
    )
    try:
        crawl = instance.create("Sensors")
        deadline = monotonic() + 3
        while len(jobs.calls) < 2 and monotonic() < deadline:
            sleep(0.02)
        # Both batches submitted; the first job is still running.
        assert len(jobs.calls) == 2
        assert all(job["status"] == "running" for job in jobs.jobs.values())
        jobs.release.set()
        final = finish(instance, crawl)
        assert final["status"] == "completed"
        assert final["counts"]["parsed"] == 4
    finally:
        jobs.release.set()
        instance.close(wait=True)


def test_partial_materials_are_extracted_again_on_request(tmp_path):
    jobs = Jobs(
        tmp_path / "jobs", statuses={"partial": "partial", "failed": "failed"}
    )

    def discovery(*args):
        return page([item("openalex", "partial"), item("openalex", "failed")])

    instance = manager(tmp_path, jobs, discoverers={"openalex": discovery})
    try:
        first = finish(instance, instance.create("Sensors"))
        jobs.statuses.update(partial="succeeded", failed="succeeded")
        retried = finish(
            instance, instance.retry_failed(first["crawl_id"], partial=True)
        )
        assert retried["counts"]["parsed"] == 2
        assert retried["counts"]["partial"] == retried["counts"]["failed"] == 0
        # Both went to a new job: the partial one was not taken from cache.
        assert sorted(payload["name"] for payload in jobs.calls[-1][1]) == [
            "failed",
            "partial",
        ]
    finally:
        instance.close(wait=True)


def test_a_stopped_crawl_can_be_deleted_and_its_materials_stay(tmp_path):
    instance = manager(
        tmp_path,
        discoverers={"openalex": lambda *args: page([item("openalex", "a")])},
    )
    try:
        crawl = finish(instance, instance.create("Sensors"))
        assert crawl["counts"]["parsed"] == 1
        assert instance.delete(crawl["crawl_id"]) == {
            "crawl_id": crawl["crawl_id"],
            "deleted": True,
        }
        assert crawl["crawl_id"] not in {
            item["crawl_id"] for item in instance.list_crawls()
        }
        # The processed material is still known: a new crawl skips it.
        again = finish(instance, instance.create("Sensors"))
        assert again["counts"]["duplicates"] + again["counts"]["parsed"] >= 1
        with pytest.raises(ValueError):
            instance.delete(instance.create("Other")["crawl_id"])
    finally:
        instance.close(wait=True)


def test_crawls_follow_the_configured_fulltext_switch(tmp_path):
    from lctrend.core.config import load_catalog

    seen = []

    class RecordingJobs(Jobs):
        def create_payloads(self, source, payloads, **kwargs):
            seen.append(kwargs.get("fulltext"))
            return super().create_payloads(source, payloads, **kwargs)

    jobs = RecordingJobs(tmp_path / "jobs")
    instance = manager(
        tmp_path,
        jobs,
        discoverers={"openalex": lambda *args: page([item("openalex", "a")])},
    )
    try:
        finish(instance, instance.create("Sensors"))
        switch = load_catalog("sources")["platforms"]["openalex"]
        assert seen == [switch["crawl_fulltext"]]
    finally:
        instance.close(wait=True)
