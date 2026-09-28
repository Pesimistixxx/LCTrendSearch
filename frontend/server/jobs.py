"""Persistent, bounded ingestion jobs for the local ingestion interface.

Job files contain progress and references, not an alternative graph
database. Original source snapshots and per-document results survive a
restart; interrupted jobs are never resumed automatically because resuming
can spend model credits.

Jobs run on one dedicated asyncio loop. Documents of a job are processed
concurrently (bounded by ``workers``); network calls (LLM, Neo4j, sources)
are awaited, CPU-bound steps and injected synchronous callables run in
worker threads. One LLM provider and one concept registry serve a job.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from concurrent.futures import Future
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from time import monotonic, sleep
from typing import Any, Callable, Iterable
from uuid import uuid4

from lctrend.core import aio
from lctrend.core.models import DocumentEnvelope, ExtractionResult
from lctrend.ingest.processed import covers, known_fulltexts, prior_inputs

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_DOCUMENT_STATUSES = {"succeeded", "partial", "failed", "cancelled"}
MODES = {"llm", "none"}
MAX_WORKERS = 16
# Progress events arrive many times per document; a 5000-document job file
# is megabytes. Progress is flushed at most this often, status changes at once.
SAVE_INTERVAL_SECONDS = 1.0


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
        "pdf_support": "Для обработки PDF требуется установленный Docling.",
        "discovery": (
            "Не удалось получить следующую страницу публикаций OpenAlex. "
            "Проверьте ключ API и параметры поиска."
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
    if stage == "discovery":
        source_messages = {
            "http_401": "OpenAlex отклонил ключ API. Обновите ключ источника.",
            "http_403": "Нет доступа к OpenAlex. Проверьте ключ и лимиты API.",
            "http_429": (
                "Достигнут лимит OpenAlex. Продолжите сбор после "
                "восстановления лимита API."
            ),
        }
        if code in source_messages:
            messages[stage] = source_messages[code]
    return {
        "code": code,
        "message": messages.get(
            stage, "Не удалось завершить задание загрузки."
        ),
    }


def default_workers() -> int:
    """Concurrent documents per job: LCTREND_WORKERS or runtime.json."""
    from lctrend.core.config import load_catalog

    value = os.getenv("LCTREND_WORKERS") or load_catalog("runtime").get(
        "ingestion", {}
    ).get("workers", 1)
    try:
        return max(1, min(MAX_WORKERS, int(value)))
    except (TypeError, ValueError):
        return 1


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
        self._max_active_jobs = max_active_jobs
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
        self._provider_factory = provider_factory or _provider_factory
        self._fulltext_attacher = fulltext_attacher
        self._file_parser = file_parser
        self._snapshot_writer = snapshot_writer
        self._pdf_support_checker = pdf_support_checker
        self._payload_hydrator = payload_hydrator
        self._saved_at: dict[str, float] = {}
        self._dirty: set[str] = set()
        self._started: set[str] = set()
        self._recover()
        self._loop = aio.LoopThread("ingestion-jobs")
        self._slots = None
        self._stopping = False
        self._wake = None
        self._flusher = self._loop.submit(self._flush_progress())

    def _recover(self) -> None:
        for path in self.directory.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job.get("mode") not in MODES:
                    job["mode"] = "llm"
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

    def _save(self, job: dict, force: bool = True) -> None:
        job_id = job["job_id"]
        self._count(job)
        job["updated_at"] = _now()
        now = monotonic()
        if (
            not force
            and now - self._saved_at.get(job_id, 0.0) < SAVE_INTERVAL_SECONDS
        ):
            self._dirty.add(job_id)
            return
        _write_json(self.directory / job_id / "job.json", job)
        self._saved_at[job_id] = now
        self._dirty.discard(job_id)

    async def _flush_progress(self) -> None:
        self._wake = asyncio.Event()
        while not self._stopping:
            try:
                await asyncio.wait_for(
                    self._wake.wait(), SAVE_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass
            if not self._dirty:
                continue
            with self._lock:
                for job_id in list(self._dirty):
                    job = self._jobs.get(job_id)
                    if job is None:
                        self._dirty.discard(job_id)
                        continue
                    try:
                        self._save(job)
                    except OSError as exc:
                        logger.warning(
                            "Cannot save progress of job %s: %s", job_id, exc
                        )

    def run(self, coroutine, timeout: float | None = None):
        """Run a coroutine on the job loop from synchronous code."""
        return self._loop.run(coroutine, timeout)

    @staticmethod
    def _validate(workers: int, mode: str) -> None:
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or not 1 <= workers <= MAX_WORKERS
        ):
            raise ValueError("workers must be 1..{MAX_WORKERS}")
        if mode not in MODES:
            raise ValueError("mode must be llm or none")

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
                if mode != "llm":
                    document["llm_status"] = "disabled"
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
            self._futures[job_id] = self._loop.submit(self._run(job_id))
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
        mode: str = "llm",
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
        mode: str = "llm",
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
        mode: str = "llm",
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
            if job_id not in self._started:
                # Still waiting for a slot: it never opens the graph.
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
            futures = list(self._futures.values())
        if wait:
            for future in futures:
                try:
                    future.result()
                except Exception:
                    logger.debug("Job ended with an error", exc_info=True)
            self._stopping = True
            if self._wake is not None:
                self._loop.loop.call_soon_threadsafe(self._wake.set)
            self._flusher.result()
            with self._lock:
                for job_id in list(self._dirty):
                    if job_id in self._jobs:
                        self._save(self._jobs[job_id])
            self._loop.stop(wait=True)

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
            self._save(job, force=False)

    def _update_document(
        self, job_id: str, doc_id: str, force: bool | None = None, **updates
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            document = next(
                item for item in job["documents"] if item["doc_id"] == doc_id
            )
            document.update(updates)
            if force is None:
                force = updates.get("status") in TERMINAL_DOCUMENT_STATUSES
            self._save(job, force=force)

    def _progress(self, job_id: str, doc_id: str, event: dict) -> None:
        allowed = {
            key: event[key]
            for key in ["stage", "packet_id", "model_calls", "max_model_calls"]
            if key in event
        }
        branch, status = event.get("branch"), event.get("status")
        if branch == "llm":
            if status == "running":
                allowed[f"{branch}_status"] = "running"
            elif event.get("stage") == "done" and status in {
                "succeeded",
                "partial",
                "failed",
            }:
                allowed[f"{branch}_status"] = status
        for key in ["llm_status"]:
            if event.get(key) in {"running", "succeeded", "partial", "failed"}:
                allowed[key] = event[key]
        if allowed:
            self._update_document(job_id, doc_id, force=False, **allowed)

    async def _run(self, job_id: str) -> None:
        if self._slots is None:
            self._slots = asyncio.Semaphore(self._max_active_jobs)
        async with self._slots:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["status"] not in ACTIVE_STATUSES:
                    return  # cancelled while waiting for a slot
                self._started.add(job_id)
            try:
                await self._execute(job_id)
            finally:
                with self._lock:
                    self._started.discard(job_id)

    async def _execute(self, job_id: str) -> None:
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
            store = await aio.call(self._store_factory)
            # All graph connectivity checks precede any potentially paid call.
            verify = getattr(store, "verify_connectivity", None)
            if verify is not None:
                await aio.call(verify)
            await aio.call(store.ensure_schema)
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
                await aio.call(checker)
            stage = "provider"
            self._stage(job_id, stage)
            # One provider per job: one OAuth token, one request semaphore
            # and one record of exhausted models for all its documents.
            provider = (
                await aio.call(self._provider_factory)
                if requires_models and job["mode"] == "llm"
                else None
            )
            registry = None
            if requires_models and job["mode"] != "none":
                from lctrend.extraction.resolver import ConceptRegistry

                # Read once; the job keeps it current as documents resolve.
                registry = ConceptRegistry(await aio.call(store.read_concepts))
                logger.info(
                    "Job %s: %d registry concepts", job_id, len(registry)
                )
            context = {
                "store": store,
                "provider": provider,
                "registry": registry,
                "publication": asyncio.Lock(),
                "workers": asyncio.Semaphore(job["workers"]),
            }

            stage = "discovery"
            self._stage(job_id, stage)
            if job["source"] == "files":
                batch = [
                    (record["doc_id"], Path(path))
                    for record, path in zip(job["documents"], task["paths"])
                ]
                await self._process_batch(job_id, batch, context)
            elif "payloads" in task:
                batch = [
                    (record["doc_id"], payload)
                    for record, payload in zip(
                        job["documents"], task["payloads"]
                    )
                ]
                await self._process_batch(job_id, batch, context)
            else:
                await self._openalex(job_id, task, context)
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
                )
            else:
                # Source exceptions may carry URLs with API credentials.
                logger.error(
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
                    await aio.call(store.close)
                except Exception as exc:
                    logger.warning(
                        "Job %s: cannot close graph store (%s)",
                        job_id,
                        type(exc).__name__,
                    )
                    logger.debug("Graph store close traceback", exc_info=True)

    async def _openalex(self, job_id: str, task: dict, context: dict) -> None:
        if self._source_fetcher is None:
            from lctrend.ingest.connectors import fetch_openalex_page

            fetch = fetch_openalex_page
        else:
            fetch = self._source_fetcher
        cursor, seen_cursors, seen_records = "*", set(), set()
        job = self._jobs[job_id]
        pending: set[asyncio.Task] = set()
        try:
            while cursor and not self._cancel[job_id].is_set():
                with self._lock:
                    remaining = job["limit"] - len(job["documents"])
                if remaining <= 0:
                    break
                if cursor in seen_cursors:
                    raise ValueError("Repeated OpenAlex cursor")
                seen_cursors.add(cursor)
                self._stage(job_id, "discovery")
                page = await aio.call(
                    fetch,
                    task["query"],
                    cursor,
                    min(100, remaining),
                    os.getenv("OPENALEX_MAILTO"),
                    task.get("filter"),
                )
                if not isinstance(page, dict):
                    raise ValueError("Invalid OpenAlex response")
                payloads = page.get("results")
                if not isinstance(payloads, list):
                    raise ValueError("Invalid OpenAlex results")
                if not payloads:
                    break
                meta = page.get("meta")
                if not isinstance(meta, dict) or "next_cursor" not in meta:
                    raise ValueError("Missing OpenAlex next cursor")
                next_cursor = meta["next_cursor"]
                if next_cursor is not None and (
                    not isinstance(next_cursor, str)
                    or not next_cursor
                    or next_cursor in seen_cursors
                ):
                    raise ValueError("Invalid or repeated OpenAlex cursor")
                if any(not isinstance(payload, dict) for payload in payloads):
                    raise ValueError("Invalid OpenAlex record")
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
                        if job["mode"] != "llm":
                            record["llm_status"] = "disabled"
                        job["documents"].append(record)
                        batch.append((doc_id, payload))
                    self._save(job)
                # Discovery of the next page overlaps with processing of this
                # one; the worker semaphore still bounds document concurrency.
                pending |= self._schedule(job_id, batch, context)
                pending = {task for task in pending if not task.done()}
                cursor = next_cursor
            if pending:
                await asyncio.gather(*pending)
        except BaseException:
            for waiting in pending:
                waiting.cancel()
            raise
        with self._lock:
            job["discovery_finished"] = not self._cancel[job_id].is_set()
            self._save(job)

    def _schedule(
        self, job_id: str, batch: list, context: dict
    ) -> set[asyncio.Task]:
        async def one(doc_id, source):
            async with context["workers"]:
                # Serialize the start decision with cancel_job so no new
                # document can slip in after cancellation has been recorded.
                with self._lock:
                    if self._cancel[job_id].is_set():
                        return
                await self._process_document(job_id, doc_id, source, context)

        return {
            asyncio.create_task(one(doc_id, source))
            for doc_id, source in batch
        }

    async def _process_batch(
        self, job_id: str, batch: list, context: dict
    ) -> None:
        """Process up to ``workers`` documents at a time.

        Cancellation stops new starts; running documents finish.
        """
        tasks = self._schedule(job_id, batch, context)
        if tasks:
            await asyncio.gather(*tasks)

    async def _process_document(
        self, job_id: str, doc_id: str, source, context: dict
    ) -> None:
        stage = "parse"
        store = context["store"]
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
                        for key in ["llm_status"]
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
                await self._publish_result(
                    job_id, doc_id, document, result, context
                )
                return
            if job["source"] == "files":
                if self._file_parser is None:
                    from lctrend.ingest.file_adapters import parse_file

                    parser = parse_file
                else:
                    parser = self._file_parser
                # Docling and the other parsers are CPU-bound.
                document = await aio.call(parser, source)
            else:
                from lctrend.ingest.adapters import (
                    parse_github,
                    parse_openalex,
                    parse_pypi,
                )

                if self._payload_hydrator is not None:
                    source = await aio.call(
                        self._payload_hydrator, job["source"], source
                    )
                elif job["source"] == "github" and "commit" not in source:
                    from lctrend.ingest.connectors import fetch_github

                    source = await fetch_github(
                        source.get("full_name")
                        or source["repository"]["full_name"],
                        os.getenv("GITHUB_TOKEN"),
                    )
                elif job["source"] == "pypi" and "info" not in source:
                    from lctrend.ingest.connectors import fetch_pypi

                    source = await fetch_pypi(source["name"])

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
                document = await aio.call(writer, document, raw)
                prior = await self._prior_inputs(job, document, store)
                if (
                    prior
                    and not job["fulltext"]
                    and covers(prior, document)
                ):
                    await self._record_metrics(document, store)
                    await self._skip(job_id, doc_id, document)
                    return
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
                    known = known_fulltexts(prior)
                    attached = await aio.call(
                        attacher,
                        document,
                        source,
                        **({"known_sha256": known} if known else {}),
                    )
                    if attached is not None:
                        document = attached
                    if prior and covers(prior, document):
                        await self._record_metrics(document, store)
                        await self._skip(job_id, doc_id, document)
                        return
            document = DocumentEnvelope.model_validate(document)
            if job["source"] == "files" and covers(
                await self._prior_inputs(job, document, store), document
            ):
                await self._skip(job_id, doc_id, document)
                return
            self._update_document(
                job_id,
                doc_id,
                title=document.title,
                source_id=document.source.record_id,
                source_coverage=document.coverage,
                stage="processing",
            )
            stage = "processing"
            if self._document_processor is None:
                from lctrend.extraction.processing import process_material

                processor = process_material
            else:
                processor = self._document_processor
            registry = context["registry"]
            context_options = {}
            if self._document_processor is None:
                context_options["context_reader"] = getattr(
                    context["store"], "read_related_chunks", None
                )
            result = await aio.call(
                processor,
                document,
                mode=job["mode"],
                provider=context["provider"],
                registry=registry if registry is not None else [],
                event=lambda event: self._progress(job_id, doc_id, event),
                **context_options,
            )
            result = ExtractionResult.model_validate(result)
            await self._publish_result(
                job_id, doc_id, document, result, context
            )
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
                    for key in ["llm_status"]
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

    async def _prior_inputs(self, job, document, store) -> list:
        """Inputs of complete extractions of this version in Neo4j."""
        if job["mode"] == "none":
            return []
        try:
            return await prior_inputs(store, document.document_version_id)
        except Exception as exc:
            # The check only saves money; it never blocks processing.
            logger.debug("Processed-version lookup failed: %s", exc)
            return []

    async def _record_metrics(self, document, store) -> None:
        """Counters of a skipped version are a new dated observation."""
        recorder = getattr(store, "record_metrics", None)
        if recorder is None:
            return
        try:
            await aio.call(recorder, document)
        except Exception as exc:
            logger.warning(
                "Metrics of %s not recorded: %s",
                document.document_version_id,
                type(exc).__name__,
            )

    async def _skip(self, job_id, doc_id, document) -> None:
        await asyncio.to_thread(
            _write_json,
            self.directory / job_id / "results" / f"{doc_id}.json",
            {
                "document": document.model_dump(mode="json"),
                "extraction": None,
                "skipped": "already_processed",
            },
        )
        logger.info(
            "Job %s document %s already processed; skipped",
            job_id,
            doc_id,
        )
        self._update_document(
            job_id,
            doc_id,
            title=document.title,
            source_id=document.source.record_id,
            source_coverage=document.coverage,
            status="succeeded",
            stage="already_processed",
            llm_status="skipped",
            result_ready=True,
            finished_at=_now(),
        )

    async def _publish_result(self, job_id, doc_id, document, result, context):
        stage = "storage"
        job = self._jobs[job_id]
        store = context["store"]
        try:
            no_text = (
                job["mode"] == "llm"
                and not document.chunks
                and result.run.metadata.get("model_calls") == 0
            )
            if job["mode"] != "llm":
                llm_status = "disabled"
            elif no_text:
                llm_status = "no_text"
            else:
                llm_status = result.run.status
            body = {
                "document": document.model_dump(mode="json"),
                "extraction": result.model_dump(mode="json"),
            }
            await asyncio.to_thread(
                _write_json,
                self.directory / job_id / "results" / f"{doc_id}.json",
                body,
            )
            self._update_document(
                job_id,
                doc_id,
                force=True,
                result_ready=True,
                stage="publication",
                assertions_count=len(result.assertions),
                entities_count=len(result.concepts),
                mentions_count=len(result.mentions),
                llm_status=llm_status,
                coverage=result.run.metadata.get(
                    "coverage", {"source": document.coverage}
                ),
            )
            stage = "publication"
            # One writer at a time keeps concept MERGEs of concurrent
            # documents from deadlocking in Neo4j.
            async with context["publication"]:
                if job["mode"] == "none":
                    await aio.call(store.write_document, document)
                else:
                    await aio.call(store.write_processed, document, result)
            status = (
                result.run.status
                if result.run.status in {"succeeded", "partial", "failed"}
                else "failed"
            )
            if status != "failed":
                error = None
            elif no_text:
                error = {
                    "code": "no_text",
                    "message": (
                        "У материала нет доступного текста для извлечения. "
                        "LLM не вызывалась."
                    ),
                }
            else:
                error = {
                    "code": "extraction_failed",
                    "message": (
                        "Извлечение не завершилось. "
                        "Диагностика сохранена в результате."
                    ),
                }
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
