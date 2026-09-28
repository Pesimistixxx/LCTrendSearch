"""Local ingestion controls and the graph TOP-15 search.

Trends are selected and ranked in ``lctrend.ranking.scoring``; this module
only serves them.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Literal, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lctrend.core import aio
from lctrend.core.config import load_catalog, load_environment

from .jobs import MAX_WORKERS, _provider_factory, default_workers

logger = logging.getLogger(__name__)


class CollectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=5000)
    workers: Optional[int] = Field(default=None, ge=1, le=MAX_WORKERS)
    mode: Literal["llm", "none"] = "llm"
    fulltext: bool = True
    filter: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("query")
    @classmethod
    def real_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Укажите направление сбора")
        return value.strip()


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["openai_compatible", "gigachat"] = "openai_compatible"
    model: str = Field(default="", max_length=200)
    base_url: str = Field(default="", max_length=1000)
    api_key: Optional[str] = Field(default=None, max_length=20000)


class SourceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    openalex_api_key: Optional[str] = Field(default=None, max_length=20000)
    openalex_mailto: str = Field(default="", max_length=320)

    @field_validator("openalex_api_key")
    @classmethod
    def clean_key(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and any(char in value for char in "\r\n\0"):
            raise ValueError("Ключ должен быть одной строкой")
        return value.strip() if value is not None else None

    @field_validator("openalex_mailto")
    @classmethod
    def clean_mailto(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Укажите email для OpenAlex")
        return value


class CrawlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str = Field(default="", max_length=1000)
    # Materials per direction and search source; None follows the full
    # source listing.
    limit: Optional[int] = Field(default=None, ge=1, le=10000)

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value: str) -> str:
        return value.strip()


class TopicRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: str = Field(min_length=1, max_length=500)
    count: int = Field(default=10, ge=1, le=30)
    # Queue a crawl per proposed topic right away (else only propose).
    queue: bool = False
    limit: Optional[int] = Field(default=None, ge=1, le=10000)

    @field_validator("direction")
    @classmethod
    def clean_direction(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("direction must not be empty")
        return value


# The UI is local: a request naming another host is a DNS-rebinding or
# proxy trick. Origin checks compare with the Host header, so the Host
# itself must be trusted first.
DEFAULT_ALLOWED_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")


def _allowed_hosts(configured=None) -> list:
    if configured is None:
        configured = [
            item.strip()
            for item in os.getenv("LCTREND_ALLOWED_HOSTS", "").split(",")
            if item.strip()
        ]
    return list(dict.fromkeys([*DEFAULT_ALLOWED_HOSTS, *configured]))


def _llm_endpoint(provider: str, base_url: str = "") -> tuple:
    """Provider and API host a stored key would be sent to."""
    catalog = load_catalog("llm")
    profile = catalog.get("gigachat", {}) if provider == "gigachat" else {}
    url = base_url or profile.get("base_url") or catalog.get("base_url", "")
    return provider, (urlsplit(url).hostname or "").casefold()


# At most this many files per upload request; the body limit follows.
MAX_UPLOAD_FILES = 100
# Multipart headers and form fields on top of the files themselves.
UPLOAD_OVERHEAD_BYTES = 1024 * 1024


def _upload_ttl_seconds() -> float:
    try:
        hours = float(os.getenv("LCTREND_UPLOAD_TTL_HOURS", 24))
    except ValueError:
        hours = 24.0
    return max(0.0, hours) * 3600


def _sweep_uploads(root: Path, active: set, ttl: float) -> int:
    """Delete upload folders older than ``ttl`` that no active job reads.

    Parsed files keep a content-addressed raw snapshot, so a finished
    upload folder is only a second copy (G-5).
    """
    import shutil
    import time

    if not root.is_dir():
        return 0
    removed = 0
    cutoff = time.time() - ttl
    for folder in root.iterdir():
        try:
            if not folder.is_dir() or folder.stat().st_mtime > cutoff:
                continue
            if any(str(path.resolve()) in active for path in folder.iterdir()):
                continue
            shutil.rmtree(folder)
            removed += 1
        except OSError as exc:
            logger.warning("Cannot remove old upload %s: %s", folder, exc)
    return removed


def _shutdown_grace() -> float:
    """Seconds a stopping server waits for running documents.

    Docker stops a container after 10 s by default, so the default fits.
    """
    try:
        return max(0.0, float(os.getenv("LCTREND_SHUTDOWN_GRACE_SECONDS", 8)))
    except ValueError:
        return 8.0


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ImportError):
        return False


def _source_status() -> dict:
    has_key = bool(os.getenv("OPENALEX_API_KEY", "").strip())
    return {
        "openalex": {
            "configured": has_key,
            "has_key": has_key,
            "mailto": os.getenv("OPENALEX_MAILTO", ""),
            "message": (
                "Ключ настроен; доступ к OpenAlex проверяется при сборе"
                if has_key
                else "Без ключа — ограниченный доступ; ключ увеличивает лимит"
            ),
        },
    }


async def _read_graph() -> dict:
    """The whole dated graph for the search ranking."""
    from lctrend.graph.store import GraphStore

    load_environment()
    async with GraphStore(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "change-me-now"),
    ) as store:
        return await store.read_temporal_data()


# The page polls readiness every few seconds; a fresh Neo4j driver per poll
# cost a TCP+TLS handshake each time (G-7). The result is reused briefly.
NEO4J_STATUS_TTL_SECONDS = 15.0
_neo4j_status: dict = {}


def _neo4j_ready() -> dict:
    key = (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
    )
    cached = _neo4j_status.get(key)
    if cached and time.monotonic() - cached[0] < NEO4J_STATUS_TTL_SECONDS:
        return dict(cached[1])
    neo = {"available": False, "message": "Neo4j недоступен"}
    try:
        from lctrend.graph.store import GraphStore

        async def check():
            async with GraphStore(
                os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "change-me-now"),
            ) as store:
                await store.verify_connectivity()

        # Sync endpoints run in a worker thread without an event loop.
        aio.run_sync(check())
        neo.update(available=True, message="Neo4j подключён")
    except Exception as exc:
        # Polled every few seconds by the page: keep it out of the console.
        logger.debug("Neo4j status check failed: %s", exc)
    _neo4j_status.clear()
    _neo4j_status[key] = (time.monotonic(), dict(neo))
    return neo


def _status() -> dict:
    """Read-only readiness; never return secrets or call a language model."""
    load_environment()
    neo = _neo4j_ready()
    catalog = load_catalog("llm")
    provider_name = os.getenv(
        "LLM_PROVIDER", catalog.get("provider", "openai_compatible")
    )
    profile = (
        catalog.get("gigachat", {}) if provider_name == "gigachat" else catalog
    )
    endpoint = os.getenv(
        "GIGACHAT_BASE_URL" if provider_name == "gigachat" else "LLM_BASE_URL"
    ) or profile.get("base_url", catalog.get("base_url", ""))
    llm = {
        "configured": False,
        "provider": provider_name,
        "base_url": endpoint,
        "models": {},
        "has_key": bool(
            os.getenv(
                "GIGACHAT_CREDENTIALS"
                if provider_name == "gigachat"
                else "LLM_API_KEY"
            )
            # A key pool file replaces the single GigaChat key.
            or provider_name == "gigachat"
            and os.getenv("GIGACHAT_KEYS_FILE")
        ),
        "message": "Настройте подключение модели",
    }
    try:
        from lctrend.llm.client import JsonLLM

        provider = JsonLLM.from_environment()
        llm.update(
            configured=True,
            models=provider.models,
            # Task routes pick models automatically; the settings form must
            # not pin the current extract model by saving it back.
            routes=bool(getattr(provider, "short_packet_chars", 0)),
            base_url=provider.base_url,
            message=(
                "Подключение настроено; доступ к модели проверяется "
                "при запуске"
            ),
        )
    except Exception as exc:
        # An unconfigured model is a normal state of a fresh checkout.
        logger.debug("LLM status check failed: %s", exc)
    return {
        "neo4j": neo,
        "llm": llm,
        "pdf": {"installed": _installed("docling")},
        "sources": _source_status(),
        "defaults": {
            "mode": "llm",
            "workers": default_workers(),
            "max_workers": MAX_WORKERS,
            "limit": 10,
            "crawl_limit": 50,
        },
        "directions": load_catalog("sources").get("crawl_directions", []),
        "server": {"local": True},
    }


def create_app(
    manager=None,
    *,
    crawl_manager=None,
    frontend_dir=None,
    upload_root=None,
    status_reader=None,
    environment_path=None,
    search_service=None,
    allowed_hosts=None,
) -> FastAPI:
    root = Path(__file__).resolve().parents[2]
    frontend = (
        Path(frontend_dir) if frontend_dir else root / "frontend" / "dist"
    )
    uploads = Path(
        upload_root
        or os.getenv("LCTREND_UPLOAD_DIR", "artifacts/ingestion/uploads")
    ).resolve()
    env_path = Path(
        environment_path
        or os.getenv("LCTREND_SETTINGS_FILE")
        or Path.cwd() / ".env"
    )
    settings_lock = RLock()

    @asynccontextmanager
    async def lifespan(app):
        load_environment()
        from lctrend.core.catalog_validation import validate_catalogs

        validate_catalogs()
        if app.state.manager is None:
            from .jobs import JobManager

            app.state.manager = JobManager()
            logger.info(
                "Job manager started in %s", app.state.manager.directory
            )
        if app.state.crawls is None and manager is None:
            from .crawl import CrawlManager

            app.state.crawls = CrawlManager(job_manager=app.state.manager)
        yield
        if crawl_manager is None and app.state.crawls is not None:
            app.state.crawls.close(wait=False)
        if manager is None:
            # Documents in flight may finish (their model calls are paid);
            # past the grace period their jobs are recorded as interrupted.
            grace = _shutdown_grace()
            logger.info("Stopping job manager (grace %.0f s)", grace)
            await asyncio.to_thread(
                app.state.manager.close, wait=True, timeout=grace
            )

    app = FastAPI(title="LCTrend: загрузка материалов", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=_allowed_hosts(allowed_hosts)
    )
    app.state.manager = manager
    app.state.crawls = crawl_manager
    app.state.search = search_service
    readiness = status_reader or _status

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path in (
            "/api/ingest/settings",
            "/api/ingest/sources/settings",
        ):
            # Pydantic's default errors echo invalid input and passwords.
            return JSONResponse(
                {"detail": "Проверьте параметры настроек"}, status_code=422
            )
        from fastapi.exception_handlers import (
            request_validation_exception_handler,
        )

        return await request_validation_exception_handler(request, exc)

    @app.middleware("http")
    async def upload_size(request: Request, call_next):
        # Reject an oversized upload before its body is received and spooled
        # to disk; per-file limits still apply while it is written (G-5).
        if request.method == "POST" and request.url.path == (
            "/api/ingest/uploads"
        ):
            limit = (
                MAX_UPLOAD_FILES
                * load_catalog("pipeline")["file_limits"]["max_file_bytes"]
                + UPLOAD_OVERHEAD_BYTES
            )
            try:
                length = int(request.headers.get("content-length", ""))
            except ValueError:
                length = None
            if length is None:
                return JSONResponse(
                    {"detail": "Не указан размер загрузки"}, status_code=411
                )
            if length > limit:
                return JSONResponse(
                    {"detail": "Загрузка превышает допустимый размер"},
                    status_code=413,
                )
        return await call_next(request)

    @app.middleware("http")
    async def local_mutations(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            parsed = urlsplit(origin)
            own = urlsplit(str(request.base_url))
            same = (parsed.scheme, parsed.hostname, parsed.port) == (
                own.scheme,
                own.hostname,
                own.port,
            )
            dev = (
                parsed.scheme == "http"
                and parsed.hostname in ("127.0.0.1", "localhost")
                and parsed.port in (5173, 5188)
            )
            if not (same or dev):
                logger.warning(
                    "Rejected %s %s from origin %s",
                    request.method,
                    request.url.path,
                    origin,
                )
                return JSONResponse(
                    {"detail": "Запуск доступен из локального интерфейса"},
                    status_code=403,
                )
        return await call_next(request)

    def get_manager():
        if app.state.manager is None:
            raise HTTPException(503, "Очередь ещё запускается")
        return app.state.manager

    def get_crawls():
        if app.state.crawls is None:
            raise HTTPException(503, "Обход источников ещё запускается")
        return app.state.crawls

    def known(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except (KeyError, FileNotFoundError) as exc:
            logger.debug("Not found: %s", exc)
            raise HTTPException(
                404, "Задание или результат не найден"
            ) from None
        except ValueError as exc:
            logger.warning("Rejected ingestion request: %s", exc)
            raise HTTPException(400, "Проверьте параметры загрузки") from None

    def require_idle(message: str):
        processing_jobs = any(
            j["status"] in ("queued", "running", "cancelling")
            for j in get_manager().list_jobs()
        )
        processing_crawls = app.state.crawls is not None and any(
            c["status"] in ("queued", "running", "pausing")
            for c in app.state.crawls.list_crawls()
        )
        if processing_jobs or processing_crawls:
            raise HTTPException(409, message)

    def save_environment(values: dict, validate=None):
        from dotenv import set_key

        before = {key: os.environ.get(key) for key in values}
        temporary = None
        try:
            os.environ.update(values)
            if validate is not None:
                validate()
            env_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=env_path.parent,
                prefix=".env-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                if env_path.exists():
                    handle.write(env_path.read_text(encoding="utf-8-sig"))
            for key, value in values.items():
                set_key(str(temporary), key, value, quote_mode="always")
            temporary.replace(env_path)
        except Exception:
            for key, old in before.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
            raise
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)

    @app.get("/api/search")
    async def search_signals(
        q: str = Query(max_length=200),
        snapshot: Optional[str] = Query(default=None, alias="date"),
    ):
        query = q.strip()
        if not query:
            raise HTTPException(422, "Введите запрос")
        try:
            cutoff = date.fromisoformat(snapshot) if snapshot else None
        except ValueError:
            raise HTTPException(422, "date: ожидается YYYY-MM-DD") from None
        if app.state.search is None:
            from lctrend.ranking.search import SearchService

            app.state.search = SearchService(_read_graph)
        try:
            return await app.state.search.search(query, cutoff)
        except Exception:
            # Connection errors name hosts; keep them in the server log.
            logger.exception("Search %r failed", query)
            raise HTTPException(
                503, "Граф недоступен: проверьте подключение к Neo4j"
            ) from None

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/llm/stats")
    def llm_stats():
        """Model request timing per key and stage since the server started:
        request and queue percentiles, output speed, keys busy now."""
        from lctrend.llm.stats import STATS

        return STATS.snapshot()

    @app.get("/api/ingest/status")
    def status():
        return readiness()

    @app.get("/api/ingest/crawls")
    def list_crawls():
        return {"crawls": get_crawls().list_crawls()}

    @app.post("/api/ingest/crawls", status_code=202)
    def create_crawl(body: CrawlRequest):
        with settings_lock:
            return known(
                get_crawls().create, topic=body.topic, limit=body.limit
            )

    @app.post("/api/ingest/topics/suggest")
    async def suggest_crawl_topics(body: TopicRequest):
        """The model proposes search topics; optionally queue them.

        Crawls run one at a time, so queued topics wait their turn.
        """
        from lctrend.ingest.topics import suggest_topics
        from lctrend.llm.client import LLMError

        crawls = get_crawls()
        exclude = [item.get("topic") or "" for item in crawls.list_crawls()]
        factory = getattr(app.state.manager, "_provider_factory", None)
        try:
            provider = await aio.call(factory or _provider_factory)
            topics = await suggest_topics(
                provider, body.direction, body.count, exclude
            )
        except LLMError as exc:
            logger.warning("Topic suggestion failed: %s", exc.code)
            raise HTTPException(
                409 if exc.code == "configuration" else 502,
                f"Модель не предложила темы ({exc.code})",
            ) from None
        created = []
        if body.queue:
            with settings_lock:
                for topic in topics:
                    crawl = known(
                        crawls.create, topic=topic["query"], limit=body.limit
                    )
                    created.append(crawl["crawl_id"])
        return {"topics": topics, "crawl_ids": created}

    @app.get("/api/ingest/crawls/{crawl_id}")
    def crawl(crawl_id: str):
        return known(get_crawls().get_crawl, crawl_id)

    @app.post("/api/ingest/crawls/{crawl_id}/pause")
    def pause_crawl(crawl_id: str):
        return known(get_crawls().pause, crawl_id)

    @app.post("/api/ingest/crawls/{crawl_id}/resume")
    def resume_crawl(crawl_id: str):
        with settings_lock:
            return known(get_crawls().resume, crawl_id)

    @app.post("/api/ingest/crawls/{crawl_id}/retry-failed")
    def retry_crawl_materials(crawl_id: str):
        with settings_lock:
            return known(get_crawls().retry_failed, crawl_id)

    @app.get("/api/ingest/crawls/{crawl_id}/materials")
    def crawl_materials(
        crawl_id: str,
        status: Optional[
            Literal["pending", "processing", "parsed", "partial", "failed"]
        ] = None,
        limit: int = Query(default=100, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        value = known(
            get_crawls().list_materials,
            crawl_id,
            status=status,
            limit=limit,
            offset=offset,
        )
        if isinstance(value, dict):
            return {
                "materials": value["items"],
                "total": value["total"],
                "limit": value["limit"],
                "offset": value["offset"],
            }
        return {"materials": value}

    @app.get("/api/ingest/jobs")
    def list_jobs():
        return {"jobs": get_manager().list_jobs()}

    @app.post("/api/ingest/jobs", status_code=202)
    def collect(body: CollectRequest):
        with settings_lock:
            values = body.model_dump()
            values["workers"] = values["workers"] or default_workers()
            return known(get_manager().create_openalex, **values)

    @app.get("/api/ingest/jobs/{job_id}")
    def job(job_id: str):
        return known(get_manager().get_job, job_id)

    @app.post("/api/ingest/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return known(get_manager().cancel_job, job_id)

    @app.get("/api/ingest/jobs/{job_id}/documents/{doc_id}/result")
    def result(job_id: str, doc_id: str):
        return known(get_manager().get_result, job_id, doc_id)

    @app.get("/api/ingest/jobs/{job_id}/documents/{doc_id}/download")
    def download(job_id: str, doc_id: str):
        value = known(get_manager().get_result, job_id, doc_id)
        return JSONResponse(
            value,
            headers={
                "Content-Disposition": 'attachment; filename="extraction.json"'
            },
        )

    @app.post("/api/ingest/uploads", status_code=202)
    async def upload(
        files: list[UploadFile] = File(...),
        mode: str = Form("llm"),
        workers: Optional[int] = Form(None),
        direction: str = Form(""),
    ):
        if (
            mode not in ("llm", "none")
            or not 1 <= (workers or default_workers()) <= MAX_WORKERS
            or not 1 <= len(files) <= MAX_UPLOAD_FILES
            or len(direction) > 1000
        ):
            raise HTTPException(
                400, "Проверьте режим, число файлов и параллельность"
            )
        catalog = load_catalog("pipeline")
        max_bytes = catalog["file_limits"]["max_file_bytes"]
        active = getattr(get_manager(), "active_inputs", set)
        await asyncio.to_thread(
            _sweep_uploads, uploads, active(), _upload_ttl_seconds()
        )
        folder = uploads / uuid4().hex
        paths = []
        try:
            for item in files:
                # Both separators must be stripped even on a different host OS.
                name = PurePosixPath(
                    PureWindowsPath(item.filename or "").name
                ).name
                name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(" .")
                if (
                    not name
                    or Path(name).suffix.casefold()
                    not in catalog["file_formats"]
                ):
                    raise HTTPException(
                        400, "Поддерживаются PDF, DOCX, TXT, Markdown и HTML"
                    )
                folder.mkdir(parents=True, exist_ok=True)
                base = Path(name)
                name = base.stem[:120] + base.suffix
                used = {p.name.casefold() for p in paths}
                suffix = 2
                while name.casefold() in used:
                    name = base.stem[:110] + f"-{suffix}" + base.suffix
                    suffix += 1
                path = folder / name
                size = 0
                with path.open("xb") as handle:
                    paths.append(path)
                    while data := await item.read(1024 * 1024):
                        size += len(data)
                        if size > max_bytes:
                            raise HTTPException(
                                413, "Файл превышает допустимый размер"
                            )
                        handle.write(data)
                if size == 0:
                    raise HTTPException(400, "Пустой файл")
            with settings_lock:
                return known(
                    get_manager().create_files,
                    paths,
                    mode=mode,
                    workers=workers or default_workers(),
                    direction=direction.strip(),
                )
        except Exception as exc:
            if isinstance(exc, HTTPException):
                logger.warning(
                    "Upload rejected (%s): %s", exc.status_code, exc.detail
                )
            else:
                logger.warning("Upload failed (%s)", type(exc).__name__)
                logger.debug("Upload failure traceback", exc_info=True)
            # These are only files created by this request, never source
            # files.
            for path in paths:
                path.unlink(missing_ok=True)
            if folder.exists():
                folder.rmdir()
            raise
        finally:
            for item in files:
                await item.close()

    @app.post("/api/ingest/settings")
    def model_settings(body: ModelSettings):
        from lctrend.llm.client import JsonLLM, LLMError

        with settings_lock:
            require_idle("Дождитесь завершения обработки перед сменой модели")
            values = {
                "LLM_PROVIDER": body.provider,
                "LLM_MODEL": body.model.strip(),
                "LLM_EXTRACT_MODEL": "",
                "LLM_REVIEW_MODEL": "",
                "GIGACHAT_BASE_URL"
                if body.provider == "gigachat"
                else "LLM_BASE_URL": body.base_url.strip(),
            }
            key_name = (
                "GIGACHAT_CREDENTIALS"
                if body.provider == "gigachat"
                else "LLM_API_KEY"
            )
            if body.api_key is not None and body.api_key.strip():
                values[key_name] = body.api_key.strip()
            else:
                # A blank key field keeps the stored key, which must never
                # follow the settings to another provider or host.
                current_provider = os.getenv(
                    "LLM_PROVIDER",
                    load_catalog("llm").get("provider", "openai_compatible"),
                )
                current = _llm_endpoint(
                    current_provider,
                    os.getenv(
                        "GIGACHAT_BASE_URL"
                        if current_provider == "gigachat"
                        else "LLM_BASE_URL",
                        "",
                    ),
                )
                target = _llm_endpoint(body.provider, body.base_url.strip())
                if target != current and os.getenv(key_name, "").strip():
                    logger.warning(
                        "Model settings rejected: stored key not reused "
                        "for provider=%s host=%s",
                        target[0],
                        target[1] or "<default>",
                    )
                    raise HTTPException(
                        400,
                        "Введите ключ заново: адрес или провайдер модели "
                        "изменился",
                    )
            try:
                # Configuration check, no request/payment.
                save_environment(values, JsonLLM.from_environment)
            except Exception as exc:
                if isinstance(exc, LLMError):
                    # LLMError carries only a code and a safe message.
                    logger.warning("Model settings rejected: %s", exc)
                    raise HTTPException(
                        400, "Проверьте модель, адрес API и ключ подключения"
                    ) from None
                logger.exception("Cannot save model settings")
                raise HTTPException(
                    500, "Не удалось сохранить настройки проекта"
                ) from None
            logger.info(
                "Model settings saved: provider=%s model=%s key_updated=%s",
                body.provider,
                values["LLM_MODEL"] or "<default>",
                body.api_key is not None and bool(body.api_key.strip()),
            )
        return readiness()

    @app.post("/api/ingest/sources/settings")
    def source_settings(body: SourceSettings):
        with settings_lock:
            require_idle(
                "Дождитесь завершения обработки перед сменой источников"
            )
            values = {"OPENALEX_MAILTO": body.openalex_mailto}
            if body.openalex_api_key:
                values["OPENALEX_API_KEY"] = body.openalex_api_key
            try:
                save_environment(values)
            except Exception:
                logger.error("Cannot save source settings")
                raise HTTPException(
                    500, "Не удалось сохранить настройки источников"
                ) from None
            logger.info(
                "OpenAlex settings saved: key_updated=%s",
                bool(body.openalex_api_key),
            )
        return {**readiness(), "sources": _source_status()}

    if (frontend / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(frontend / "assets")),
            name="assets",
        )

    @app.get("/")
    def page():
        path = frontend / "index.html"
        if not path.is_file():
            raise HTTPException(
                503, "Интерфейс ещё не собран: выполните сборку frontend"
            )
        return FileResponse(path)

    @app.get("/ingest.html")
    def ingestion_page():
        return RedirectResponse("/#view=ingest")

    return app


def serve(host: str = "127.0.0.1", port: int = 5188) -> None:
    import uvicorn

    from lctrend.core.logging_config import setup_logging

    log_file = setup_logging(loggers=("lctrend", "frontend.server"))
    logger.info("Starting ingestion server on http://%s:%s", host, port)
    logger.debug("Writing detailed log to %s", log_file)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
