"""Persistent, bounded ingestion jobs for the local ingestion interface.

Job files contain progress and references, not an alternative graph
database. Original source snapshots and per-document results survive a
restart; interrupted jobs are never resumed automatically because resuming
can spend model credits.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from time import sleep
from typing import Any, Callable, Iterable
from uuid import uuid4

from lctrend.core.models import DocumentEnvelope, ExtractionResult

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_DOCUMENT_STATUSES = {"succeeded", "partial", "failed", "cancelled"}
MODES = {"hybrid", "llm", "gliner", "none"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    """Publish a complete JSON file atomically on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=".job-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value, output, ensure_ascii=False, indent=2, allow_nan=False
            )
            output.flush()
            os.fsync(output.fileno())
        # Windows readers and antivirus can briefly hold the destination file.
        # Retry that sharing error briefly; surface persistent failure.
        for attempt, delay in enumerate((0, 0.05, 0.1, 0.2)):
            if delay:
                sleep(delay)
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
    finally:
        temporary.unlink(missing_ok=True)


def _error(exc: Exception, stage: str) -> dict[str, str]:
    # Exceptions can contain request URLs, authorization headers or server
    # bodies. Keep only a bounded identifier and a message controlled by this
    # application.
    code = str(getattr(exc, "code", type(exc).__name__))
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", code):
        code = "processing_failed"
    messages = {
        "neo4j": (
            "Не удалось подключиться к Neo4j. "
            "Проверьте настройки и запуск базы."
        ),
        "provider": (
            "Не удалось подготовить LLM. Проверьте настройки провайдера."
        ),
        "model_loading": (
            "Не удалось загрузить GLiNER. "
            "Проверьте зависимости и доступность модели."
        ),
        "pdf_support": "Для обработки PDF требуется установленный Docling.",
        "discovery": (
            "Не удалось получить следующую страницу публикаций OpenAlex."
        ),
        "parse": "Не удалось разобрать исходный документ.",
        "fulltext": "Не удалось подготовить полный текст публикации.",
        "processing": (
            "Ошибка обработки документа. "
            "Подробности этапов сохранены в результате."
        ),
        "publication": (
            "Результат сохранён локально, но запись в Neo4j не завершилась."
        ),
        "storage": "Не удалось сохранить результат обработки на диск.",
    }
    return {
        "code": code,
        "message": messages.get(
            stage, "Не удалось завершить задание загрузки."
        ),
    }


def _store_factory():
    from lctrend.core.config import load_environment
    from lctrend.graph.store import GraphStore

    load_environment()
    return GraphStore(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "change-me-now"),
    )


def _provider_factory():
    from lctrend.llm.client import JsonLLM

    return JsonLLM.from_environment()


def _ner_runtime_factory():
    from lctrend.extraction.processing import NerRuntime

    return NerRuntime(model_name=os.getenv("GLINER_MODEL"))


class JobManager:
    """Run ingestion in background with explicit per-document progress.

    Dependencies are injectable so an offline test never needs Neo4j, a model
    download or a paid LLM request. The default publisher always uses Neo4j.
    """

    def __init__(
        self,
        directory: Path | str = "artifacts/ingestion/jobs",
        *,
        max_active_jobs: int = 1,
        source_fetcher: Callable | None = None,
        document_processor: Callable | None = None,
        store_factory: Callable | None = None,
        ner_runtime_factory: Callable | None = None,
        provider_factory: Callable | None = None,
        fulltext_attacher: Callable | None = None,
        file_parser: Callable | None = None,
        snapshot_writer: Callable | None = None,
        pdf_support_checker: Callable | None = None,
        payload_hydrator: Callable | None = None,
        max_cached_jobs: int = 100,
    ):
        if not 1 <= max_active_jobs <= 4:
            raise ValueError("max_active_jobs must be 1..4")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._publication_lock = RLock()
        self._jobs: dict[str, dict] = {}
        self._tasks: dict[str, dict] = {}
        self._cancel: dict[str, Event] = {}
        self._futures: dict[str, Future] = {}
        self._closed = False
        if not 1 <= max_cached_jobs <= 1000:
            raise ValueError("max_cached_jobs must be 1..1000")
        self._max_cached_jobs = max_cached_jobs
        self._source_fetcher = source_fetcher
        self._document_processor = document_processor
        self._store_factory = store_factory or _store_factory
        self._ner_runtime_factory = ner_runtime_factory or _ner_runtime_factory
        self._provider_factory = provider_factory or _provider_factory
        self._fulltext_attacher = fulltext_attacher
        self._file_parser = file_parser
        self._snapshot_writer = snapshot_writer
        self._pdf_support_checker = pdf_support_checker
        self._payload_hydrator = payload_hydrator
        self._shared_ner_runtime = None
        self._recover()
        self._pool = ThreadPoolExecutor(
            max_workers=max_active_jobs, thread_name_prefix="ingestion-job"
        )

    def _recover(self) -> None:
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("mode") == "both":
                    job["mode"] = "hybrid"
                job_id = job["job_id"]
                if job_id != path.parent.name or not re.fullmatch(
                    r"[a-f0-9]{32}", job_id
                ):
                    continue
                if job["status"] in ACTIVE_STATUSES:
                    finished = _now()
                    job.update(
                        status="interrupted",
                        stage="interrupted",
                        finished_at=finished,
                        error={
                            "code": "interrupted",
                            "message": (
                                "Задание прервано перезапуском. "
                                "Автоматический повтор LLM отключён."
                            ),
                        },
                    )
                    for document in job.get("documents", []):
                        if document["status"] in {"queued", "running"}:
                            document.update(
                                interrupted_stage=document.get("stage"),
                                status="failed"
                                if document["status"] == "running"
                                else "cancelled",
                                stage="interrupted",
                                finished_at=finished,
                                error={
                                    "code": "interrupted",
                                    "message": (
                                        "Обработка была прервана перезапуском."
                                    ),
                                },
                            )
                    self._count(job)
                    job["updated_at"] = finished
                    _write_json(path, job)
                    logger.warning(
                        "Job %s was interrupted by a restart", job_id
                    )
                self._jobs[job_id] = job
                self._cancel[job_id] = Event()
                self._prune()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                # A damaged unrelated file must not erase other recoverable
                # jobs.
                logger.warning(
                    "Skipping unreadable job file %s: %s", path, exc
                )
                continue
        if self._jobs:
            logger.info(
                "Recovered %d job(s) from %s", len(self._jobs), self.directory
            )

    @staticmethod
    def _count(job: dict) -> None:
        documents = job["documents"]
        counts = {
            status: sum(d["status"] == status for d in documents)
            for status in [
                "queued",
                "running",
                "succeeded",
                "partial",
                "failed",
                "cancelled",
            ]
        }
        job["counts"] = {
            "total": job["limit"],
            "discovered": len(documents),
            "completed": sum(counts[s] for s in TERMINAL_DOCUMENT_STATUSES),
            **counts,
        }

    def _prune(self, *_):
        """Keep active jobs and a bounded recent terminal metadata cache."""
        with self._lock:
            terminal = sorted(
                (
                    job
                    for job in self._jobs.values()
                    if job["status"] not in ACTIVE_STATUSES
                ),
                key=lambda job: job["created_at"],
                reverse=True,
            )
            for job in terminal[self._max_cached_jobs :]:
                job_id = job["job_id"]
                future = self._futures.get(job_id)
                if future is not None and not future.done():
                    continue
                for mapping in [
                    self._jobs,
                    self._tasks,
                    self._cancel,
                    self._futures,
                ]:
                    mapping.pop(job_id, None)

    def _save(self, job: dict) -> None:
        self._count(job)
        job["updated_at"] = _now()
        _write_json(self.directory / job["job_id"] / "job.json", job)

    @staticmethod
    def _validate(workers: int, mode: str) -> None:
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or not 1 <= workers <= 4
        ):
            raise ValueError("workers must be 1..4")
        if mode not in MODES:
            raise ValueError("mode must be hybrid, llm, gliner or none")

    @staticmethod
    def _document_record(doc_id: str, title: str, source_id: str) -> dict:
        return {
            "doc_id": doc_id,
            "title": title,
            "source_id": source_id,
            "status": "queued",
            "stage": "queued",
            "coverage": None,
            "source_coverage": None,
            "llm_status": "queued",
            "gliner_status": "queued",
            "assertions_count": 0,
            "entities_count": 0,
            "mentions_count": 0,
            "error": None,
            "result_ready": False,
            "started_at": None,
            "finished_at": None,
        }

    def _create(
        self,
        source: str,
        direction: str,
        limit: int,
        workers: int,
        mode: str,
        fulltext: bool,
        task: dict,
        documents: list[dict],
        on_created: Callable | None = None,
    ) -> dict:
        self._validate(workers, mode)
        with self._lock:
            if self._closed:
                raise RuntimeError("Job manager is closed")
            job_id = uuid4().hex
            created = _now()
            job = {
                "job_id": job_id,
                "source": source,
                "direction": direction,
                "query": direction if source == "openalex" else "",
                "mode": mode,
                "fulltext": fulltext,
                "limit": limit,
                "workers": workers,
                "status": "queued",
                "stage": "queued",
                "created_at": created,
                "updated_at": created,
                "finished_at": None,
                "error": None,
                "discovery_finished": source == "files",
                "documents": documents,
            }
            for document in documents:
                if mode not in {"llm", "hybrid"}:
                    document["llm_status"] = "disabled"
                if mode not in {"gliner", "hybrid"}:
                    document["gliner_status"] = "disabled"
            _write_json(self.directory / job_id / "task.json", task)
            self._save(job)
            self._jobs[job_id] = job
            self._tasks[job_id] = task
            self._cancel[job_id] = Event()
            if on_created is not None:
                try:
                    on_created(deepcopy(job))
                except Exception as exc:
                    self._finish(job, "failed", _error(exc, "storage"))
                    raise
            self._futures[job_id] = self._pool.submit(self._run, job_id)
            self._futures[job_id].add_done_callback(self._prune)
            logger.info(
                "Job %s queued: source=%s mode=%s limit=%d workers=%d",
                job_id,
                source,
                mode,
                limit,
                workers,
            )
            return deepcopy(job)

    def create_openalex(
        self,
        query: str,
        limit: int,
        workers: int = 1,
        mode: str = "hybrid",
        fulltext: bool = True,
        filter: str | None = None,
    ) -> dict:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 4000
        ):
            raise ValueError(
                "query must be a nonempty string of at most 4000 characters"
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 5000
        ):
            raise ValueError("limit must be 1..5000")
        if filter is not None and (
            not isinstance(filter, str) or len(filter) > 4000
        ):
            raise ValueError(
                "filter must be a string of at most 4000 characters"
            )
        return self._create(
            "openalex",
            query.strip(),
            limit,
            workers,
            mode,
            bool(fulltext),
            {"query": query.strip(), "filter": filter},
            [],
        )

    def create_files(
        self,
        paths: Iterable[Path | str],
        mode: str = "hybrid",
        workers: int = 1,
        direction: str = "",
    ) -> dict:
        paths = [Path(path).expanduser().resolve() for path in paths]
        if not 1 <= len(paths) <= 100:
            raise ValueError("file count must be 1..100")
        if not isinstance(direction, str) or len(direction) > 4000:
            raise ValueError(
                "direction must be a string of at most 4000 characters"
            )
        if any(not path.is_file() for path in paths):
            raise ValueError("every input must be an existing file")
        documents = [
            self._document_record(f"d{index:06d}", path.name, path.name)
            for index, path in enumerate(paths, 1)
        ]
        return self._create(
            "files",
            direction.strip(),
            len(paths),
            workers,
            mode,
            False,
            {"paths": [str(path) for path in paths]},
            documents,
        )

    def create_payloads(
        self,
        source: str,
        payloads: Iterable[dict],
        direction: str = "",
        mode: str = "hybrid",
        fulltext: bool = True,
        workers: int = 1,
        on_created: Callable | None = None,
        cached_results: list[dict | None] | None = None,
    ) -> dict:
        """Process discovered records without repeating discovery."""
        payloads = list(payloads)
        if (
            source not in {"openalex", "github", "pypi"}
            or not 1 <= len(payloads) <= 100
        ):
            raise ValueError("Expected 1..100 supported source records")
        if not all(isinstance(payload, dict) for payload in payloads):
            raise ValueError("Source records must be objects")
        documents = []
        for index, payload in enumerate(payloads, 1):
            metadata = (
                payload.get("repository") or payload.get("info") or payload
            )
            source_id = str(
                metadata.get("id")
                or metadata.get("full_name")
                or metadata.get("name")
                or index
            )
            title = str(
                metadata.get("title")
                or metadata.get("display_name")
                or metadata.get("full_name")
                or metadata.get("name")
                or source_id
            )
            documents.append(
                self._document_record(f"d{index:06d}", title, source_id)
            )
        task = {"payloads": payloads}
        if cached_results is not None:
            if len(cached_results) != len(payloads):
                raise ValueError("Cached result count differs from records")
            from lctrend.core.models import validate_extraction

            for cached in cached_results:
                if cached is not None:
                    saved_document = DocumentEnvelope.model_validate(
                        cached["document"]
                    )
                    saved_result = ExtractionResult.model_validate(
                        cached["extraction"]
                    )
                    if saved_result.run.status not in {"succeeded", "partial"}:
                        raise ValueError(
                            "Failed extraction cannot be republished"
                        )
                    validate_extraction(saved_document, saved_result)
            task["cached_results"] = {
                record["doc_id"]: cached
                for record, cached in zip(documents, cached_results)
                if cached is not None
            }
        return self._create(
            source,
            direction,
            len(payloads),
            workers,
            mode,
            bool(fulltext and source == "openalex"),
            task,
            documents,
            on_created,
        )

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return deepcopy(
                sorted(
                    self._jobs.values(),
                    key=lambda item: item["created_at"],
                    reverse=True,
                )
            )

    def get_job(self, job_id: str) -> dict:
        with self._lock:
            if job_id in self._jobs:
                return deepcopy(self._jobs[job_id])
            if not re.fullmatch(r"[a-f0-9]{32}", job_id):
                raise KeyError(job_id)
            path = self.directory / job_id / "job.json"
            if not path.is_file():
                raise KeyError(job_id)
            job = json.loads(path.read_text(encoding="utf-8"))
            if job.get("job_id") != job_id:
                raise ValueError(
                    "Job file identity differs from its directory"
                )
            return job

    def get_result(self, job_id: str, doc_id: str) -> dict:
        with self._lock:
            job = self.get_job(job_id)
            document = next(
                (
                    item
                    for item in job["documents"]
                    if item["doc_id"] == doc_id
                ),
                None,
            )
            if document is None or not document["result_ready"]:
                raise KeyError(doc_id)
            path = self.directory / job_id / "results" / f"{doc_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def cancel_job(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                disk_job = self.get_job(job_id)
                if disk_job["status"] not in ACTIVE_STATUSES:
                    return disk_job
                raise KeyError(job_id)
            job = self._jobs[job_id]
            if job["status"] not in ACTIVE_STATUSES:
                return deepcopy(job)
            self._cancel[job_id].set()
            logger.info("Job %s cancellation requested", job_id)
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                self._finish(job, "cancelled")
            else:
                job.update(status="cancelling", stage="cancelling")
                self._save(job)
            return deepcopy(job)

    def close(self, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
            for job_id, job in self._jobs.items():
                if job["status"] in ACTIVE_STATUSES:
                    self.cancel_job(job_id)
        self._pool.shutdown(wait=wait, cancel_futures=True)

    shutdown = close

    def _finish(
        self, job: dict, status: str, error: dict | None = None
    ) -> None:
        finished = _now()
        for document in job["documents"]:
            if document["status"] == "queued":
                document.update(
                    status="cancelled", stage="cancelled", finished_at=finished
                )
                if document["llm_status"] == "queued":
                    document["llm_status"] = "cancelled"
                if document["gliner_status"] == "queued":
                    document["gliner_status"] = "cancelled"
        job.update(
            status=status, stage=status, finished_at=finished, error=error
        )
        self._save(job)
        counts = job["counts"]
        logger.info(
            "Job %s %s: %d succeeded, %d partial, %d failed, %d cancelled",
            job["job_id"],
            status,
            counts["succeeded"],
            counts["partial"],
            counts["failed"],
            counts["cancelled"],
        )
        self._prune()

    def _stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] != "cancelling":
                job["stage"] = stage
            self._save(job)

    def _update_document(self, job_id: str, doc_id: str, **updates) -> None:
        with self._lock:
            job = self._jobs[job_id]
            document = next(
                item for item in job["documents"] if item["doc_id"] == doc_id
            )
            document.update(updates)
            self._save(job)

    def _progress(self, job_id: str, doc_id: str, event: dict) -> None:
        allowed = {
            key: event[key]
            for key in ["stage", "packet_id", "model_calls", "max_model_calls"]
            if key in event
        }
        branch, status = event.get("branch"), event.get("status")
        if branch in {"llm", "gliner"}:
            if status == "running":
                allowed[f"{branch}_status"] = "running"
            elif event.get("stage") == "done" and status in {
                "succeeded",
                "partial",
                "failed",
            }:
                allowed[f"{branch}_status"] = status
        for key in ["llm_status", "gliner_status"]:
            if event.get(key) in {"running", "succeeded", "partial", "failed"}:
                allowed[key] = event[key]
        if allowed:
            self._update_document(job_id, doc_id, **allowed)

    def _run(self, job_id: str) -> None:
        stage = "neo4j"
        store = None
        try:
            with self._lock:
                job = self._jobs[job_id]
                if self._cancel[job_id].is_set():
                    self._finish(job, "cancelled")
                    return
                job.update(status="running", stage="neo4j")
                self._save(job)
                task = self._tasks[job_id]
                requires_models = "cached_results" not in task or len(
                    task["cached_results"]
                ) < len(job["documents"])
            logger.info("Job %s started", job_id)
            store = self._store_factory()
            # All graph connectivity checks precede any potentially paid call.
            verify = getattr(store, "verify_connectivity", None)
            if verify is not None:
                verify()
            store.ensure_schema()
            if self._cancel[job_id].is_set():
                with self._lock:
                    self._finish(job, "cancelled")
                return
            if (
                requires_models
                and job["source"] == "openalex"
                and job["fulltext"]
            ):
                stage = "pdf_support"
                self._stage(job_id, stage)
                if self._pdf_support_checker is None:
                    from lctrend.ingest.fulltext import require_pdf_support

                    checker = require_pdf_support
                else:
                    checker = self._pdf_support_checker
                checker()
            stage = "provider"
            self._stage(job_id, stage)
            first_provider = (
                self._provider_factory()
                if requires_models and job["mode"] in {"hybrid", "llm"}
                else None
            )
            runtime = None
            if requires_models and job["mode"] in {"hybrid", "gliner"}:
                stage = "model_loading"
                self._stage(job_id, stage)
                try:
                    with self._lock:
                        if self._shared_ner_runtime is None:
                            self._shared_ner_runtime = (
                                self._ner_runtime_factory()
                            )
                        runtime = self._shared_ner_runtime
                    prepare = getattr(runtime, "prepare", None)
                    if prepare is not None:
                        prepare()
                except Exception as exc:
                    if job["mode"] == "gliner":
                        raise
                    logger.warning(
                        "Auxiliary NER disabled (%s)", type(exc).__name__
                    )
                    runtime = None
            provider_lock = RLock()
            providers = [first_provider]

            def provider_for_document():
                if job["mode"] not in {"hybrid", "llm"}:
                    return None
                with provider_lock:
                    if providers:
                        return providers.pop()
                return self._provider_factory()

            stage = "discovery"
            self._stage(job_id, stage)
            with ThreadPoolExecutor(
                max_workers=job["workers"],
                thread_name_prefix="ingestion-document",
            ) as workers:
                if job["source"] == "files":
                    batch = [
                        (record["doc_id"], Path(path))
                        for record, path in zip(
                            job["documents"], task["paths"]
                        )
                    ]
                    self._process_batch(
                        job_id,
                        batch,
                        workers,
                        store,
                        runtime,
                        provider_for_document,
                    )
                elif "payloads" in task:
                    batch = [
                        (record["doc_id"], payload)
                        for record, payload in zip(
                            job["documents"], task["payloads"]
                        )
                    ]
                    self._process_batch(
                        job_id,
                        batch,
                        workers,
                        store,
                        runtime,
                        provider_for_document,
                    )
                else:
                    self._openalex(
                        job_id,
                        task,
                        workers,
                        store,
                        runtime,
                        provider_for_document,
                    )
            with self._lock:
                if self._cancel[job_id].is_set():
                    self._finish(job, "cancelled")
                elif job["counts"]["failed"] and not (
                    job["counts"]["succeeded"] or job["counts"]["partial"]
                ):
                    self._finish(
                        job,
                        "failed",
                        {
                            "code": "all_documents_failed",
                            "message": (
                                "Не удалось обработать ни один полученный "
                                "документ."
                            ),
                        },
                    )
                else:
                    self._finish(job, "completed")
        except Exception as exc:
            if self._cancel[job_id].is_set():
                logger.debug(
                    "Job %s stopped at stage %s after cancellation",
                    job_id,
                    stage,
                    exc_info=True,
                )
            else:
                logger.exception(
                    "Job %s failed at stage %s (%s)",
                    job_id,
                    stage,
                    _error(exc, stage)["code"],
                )
            with self._lock:
                job = self._jobs[job_id]
                self._finish(
                    job,
                    "cancelled" if self._cancel[job_id].is_set() else "failed",
                    None
                    if self._cancel[job_id].is_set()
                    else _error(exc, stage),
                )
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception as exc:
                    logger.warning(
                        "Job %s: cannot close graph store (%s)",
                        job_id,
                        type(exc).__name__,
                    )
                    logger.debug("Graph store close traceback", exc_info=True)

    def _openalex(
        self, job_id: str, task: dict, workers, store, runtime, provider
    ) -> None:
        if self._source_fetcher is None:
            from lctrend.ingest.connectors import fetch_openalex_page

            fetch = fetch_openalex_page
        else:
            fetch = self._source_fetcher
        cursor, seen_cursors, seen_records = "*", set(), set()
        job = self._jobs[job_id]
        while cursor and not self._cancel[job_id].is_set():
            with self._lock:
                remaining = job["limit"] - len(job["documents"])
            if remaining <= 0:
                break
            if cursor in seen_cursors:
                raise ValueError("Repeated OpenAlex cursor")
            seen_cursors.add(cursor)
            self._stage(job_id, "discovery")
            page = fetch(
                task["query"],
                cursor,
                min(100, remaining),
                os.getenv("OPENALEX_MAILTO"),
                task.get("filter"),
            )
            payloads = page.get("results", [])
            if not isinstance(payloads, list):
                raise ValueError("Invalid OpenAlex results")
            if not payloads:
                break
            batch = []
            with self._lock:
                for payload in payloads:
                    if not isinstance(payload, dict):
                        raise ValueError("Invalid OpenAlex record")
                    source_id = str(payload.get("id") or "")
                    if source_id and source_id in seen_records:
                        continue
                    if len(job["documents"]) >= job["limit"]:
                        break
                    if source_id:
                        seen_records.add(source_id)
                    doc_id = f"d{len(job['documents']) + 1:06d}"
                    record = self._document_record(
                        doc_id,
                        str(
                            payload.get("title")
                            or payload.get("display_name")
                            or source_id
                        ),
                        source_id,
                    )
                    if job["mode"] not in {"llm", "hybrid"}:
                        record["llm_status"] = "disabled"
                    if job["mode"] not in {"gliner", "hybrid"}:
                        record["gliner_status"] = "disabled"
                    job["documents"].append(record)
                    batch.append((doc_id, payload))
                self._save(job)
            self._process_batch(
                job_id, batch, workers, store, runtime, provider
            )
            cursor = page.get("meta", {}).get("next_cursor")
        with self._lock:
            job["discovery_finished"] = not self._cancel[job_id].is_set()
            self._save(job)

    def _process_batch(
        self, job_id: str, batch: list, workers, store, runtime, provider
    ) -> None:
        """Submit at most ``workers`` documents.

        Cancellation stops new submits.
        """
        waiting = iter(batch)
        pending: set[Future] = set()
        exhausted = False
        while pending or not exhausted:
            while (
                len(pending) < self._jobs[job_id]["workers"] and not exhausted
            ):
                # Serialize the scheduling decision with cancel_job so no new
                # document can slip in after cancellation has been recorded.
                with self._lock:
                    if self._cancel[job_id].is_set():
                        exhausted = True
                        break
                    try:
                        doc_id, source = next(waiting)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(
                        workers.submit(
                            self._process_document,
                            job_id,
                            doc_id,
                            source,
                            store,
                            runtime,
                            provider,
                        )
                    )
            if self._cancel[job_id].is_set():
                exhausted = True
            if pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()

    def _process_document(
        self,
        job_id: str,
        doc_id: str,
        source,
        store,
        runtime,
        provider_factory,
    ) -> None:
        stage = "parse"
        try:
            with self._lock:
                if self._cancel[job_id].is_set():
                    record = next(
                        item
                        for item in self._jobs[job_id]["documents"]
                        if item["doc_id"] == doc_id
                    )
                    branch_updates = {
                        key: "cancelled"
                        for key in ["llm_status", "gliner_status"]
                        if record[key] == "queued"
                    }
                    self._update_document(
                        job_id,
                        doc_id,
                        status="cancelled",
                        stage="cancelled",
                        finished_at=_now(),
                        **branch_updates,
                    )
                    return
                self._update_document(
                    job_id,
                    doc_id,
                    status="running",
                    stage=stage,
                    started_at=_now(),
                )
            job = self._jobs[job_id]
            cached = self._tasks[job_id].get("cached_results", {}).get(doc_id)
            if cached is not None:
                document = DocumentEnvelope.model_validate(cached["document"])
                result = ExtractionResult.model_validate(cached["extraction"])
                self._update_document(
                    job_id,
                    doc_id,
                    title=document.title,
                    source_id=document.source.record_id,
                    source_coverage=document.coverage,
                )
                self._publish_result(job_id, doc_id, document, result, store)
                return
            if job["source"] == "files":
                if self._file_parser is None:
                    from lctrend.ingest.file_adapters import parse_file

                    parser = parse_file
                else:
                    parser = self._file_parser
                document = parser(source)
            else:
                from lctrend.ingest.adapters import (
                    parse_github,
                    parse_openalex,
                    parse_pypi,
                )

                if self._payload_hydrator is not None:
                    source = self._payload_hydrator(job["source"], source)
                elif job["source"] == "github" and "commit" not in source:
                    from lctrend.ingest.connectors import fetch_github

                    source = fetch_github(
                        source.get("full_name")
                        or source["repository"]["full_name"],
                        os.getenv("GITHUB_TOKEN"),
                    )
                elif job["source"] == "pypi" and "info" not in source:
                    from lctrend.ingest.connectors import fetch_pypi

                    source = fetch_pypi(source["name"])

                raw = json.dumps(
                    source, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                document = {
                    "openalex": parse_openalex,
                    "github": parse_github,
                    "pypi": parse_pypi,
                }[job["source"]](source, raw=raw)
                if self._snapshot_writer is None:
                    from lctrend.ingest.snapshots import persist_snapshot

                    writer = persist_snapshot
                else:
                    writer = self._snapshot_writer
                document = writer(document, raw)
                if job["fulltext"]:
                    stage = "fulltext"
                    self._update_document(job_id, doc_id, stage=stage)
                    if self._fulltext_attacher is None:
                        from lctrend.ingest.fulltext import (
                            attach_openalex_fulltext,
                        )

                        attacher = attach_openalex_fulltext
                    else:
                        attacher = self._fulltext_attacher
                    attached = attacher(document, source)
                    if attached is not None:
                        document = attached
            document = DocumentEnvelope.model_validate(document)
            self._update_document(
                job_id,
                doc_id,
                title=document.title,
                source_id=document.source.record_id,
                source_coverage=document.coverage,
                stage="processing",
            )
            with self._publication_lock:
                registry = (
                    store.read_concepts() if job["mode"] != "none" else []
                )
            stage = "processing"
            if self._document_processor is None:
                from lctrend.extraction.processing import process_material

                processor = process_material
            else:
                processor = self._document_processor
            result = processor(
                document,
                mode=job["mode"],
                provider=provider_factory(),
                ner_runtime=runtime,
                registry=registry,
                event=lambda event: self._progress(job_id, doc_id, event),
            )
            result = ExtractionResult.model_validate(result)
            self._publish_result(job_id, doc_id, document, result, store)
        except Exception as exc:
            # Raw exception text may hold URLs or secrets; it stays in the
            # DEBUG traceback of the log file.
            logger.warning(
                "Job %s document %s failed at stage %s (%s, %s)",
                job_id,
                doc_id,
                stage,
                _error(exc, stage)["code"],
                type(exc).__name__,
            )
            logger.debug(
                "Job %s document %s traceback", job_id, doc_id, exc_info=True
            )
            with self._lock:
                record = next(
                    item
                    for item in self._jobs[job_id]["documents"]
                    if item["doc_id"] == doc_id
                )
                branch_updates = {
                    key: "not_started" if record[key] == "queued" else "failed"
                    for key in ["llm_status", "gliner_status"]
                    if record[key] in {"queued", "running"}
                }
            self._update_document(
                job_id,
                doc_id,
                status="failed",
                stage=stage,
                error=_error(exc, stage),
                finished_at=_now(),
                **branch_updates,
            )

    def _publish_result(self, job_id, doc_id, document, result, store):
        stage = "storage"
        job = self._jobs[job_id]
        try:
            llm_status = (
                result.run.status
                if job["mode"] in {"hybrid", "llm"}
                else "disabled"
            )
            gliner_status = (
                result.run.metadata.get("ner", {}).get("status", "disabled")
                if job["mode"] == "hybrid"
                else result.run.status
                if job["mode"] == "gliner"
                else "disabled"
            )
            stage = "storage"
            body = {
                "document": document.model_dump(mode="json"),
                "extraction": result.model_dump(mode="json"),
            }
            _write_json(
                self.directory / job_id / "results" / f"{doc_id}.json", body
            )
            self._update_document(
                job_id,
                doc_id,
                result_ready=True,
                stage="publication",
                assertions_count=len(result.assertions),
                entities_count=len(result.concepts),
                mentions_count=len(result.mentions),
                llm_status=llm_status,
                gliner_status=gliner_status,
                coverage=result.run.metadata.get(
                    "coverage", {"source": document.coverage}
                ),
            )
            stage = "publication"
            with self._publication_lock:
                if job["mode"] == "none":
                    store.write_document(document)
                else:
                    store.write_processed(document, result)
            status = (
                result.run.status
                if result.run.status in {"succeeded", "partial", "failed"}
                else "failed"
            )
            error = (
                None
                if status != "failed"
                else {
                    "code": "extraction_failed",
                    "message": (
                        "Извлечение не завершилось. "
                        "Диагностика сохранена в результате."
                    ),
                }
            )
            self._update_document(
                job_id,
                doc_id,
                status=status,
                stage="done",
                finished_at=_now(),
                error=error,
            )
            logger.info(
                "Job %s document %s %s: %d assertions, %d entities",
                job_id,
                doc_id,
                status,
                len(result.assertions),
                len(result.concepts),
            )
        except Exception as exc:
            self._update_document(
                job_id,
                doc_id,
                status="failed",
                stage=stage,
                error=_error(exc, stage),
                finished_at=_now(),
            )
