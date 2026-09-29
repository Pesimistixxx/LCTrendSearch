from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .core.aio import resolve
from .core.config import load_catalog, load_environment
from .core.logging_config import setup_logging
from .core.models import stable_id
from .extraction.processing import process_material, seed_semantic
from .extraction.resolver import ConceptRegistry
from .graph.subgraphs import sample_subgraph, write_subgraph_rows
from .graph.training import (
    build_dataset_rows,
    build_snapshot_rows,
    dataset_feature_fields,
    write_dataset_rows,
    write_snapshot_rows,
)
from .ingest.adapters import (
    parse_epo,
    parse_github,
    parse_openalex,
    parse_pypi,
)
from .ingest.connectors import (
    fetch_github,
    fetch_openalex,
    fetch_openalex_page,
    fetch_pypi,
    fetch_pypi_projects,
)
from .ingest.fulltext import attach_openalex_fulltext, require_pdf_support
from .ingest.processed import covers, known_fulltexts, prior_inputs
from .ingest.snapshots import persist_snapshot as _snapshot
from .linking.reconcile import reconcile_quietly
from .session import embed_concepts as _embed_concepts
from .session import open_store as _store
from .session import opened as _opened
from .session import temporal_corpus as _temporal_data
from .session import with_graph
from .taxonomy import (
    TaxonomyConcept,
    build_taxonomy,
)

logger = logging.getLogger(__name__)

EXTRACTORS = ["llm"]
PARSERS = {
    "openalex": parse_openalex,
    "github": parse_github,
    "pypi": parse_pypi,
}


def _parse(kind: str, path: Path):
    if kind == "file":
        from .ingest.file_adapters import parse_file

        return parse_file(path)
    raw = path.read_bytes()
    if kind == "epo":
        document = parse_epo(
            raw.decode("utf-8"),
            path.resolve().as_uri(),
            datetime.now(timezone.utc).isoformat(),
        )
    else:
        payload = json.loads(raw)
        document = PARSERS[kind](payload, raw=raw)
    return _snapshot(document, raw)


async def _write_ingested_async(
    document,
    extract: bool,
    store,
    extractor: str = "llm",
    provider=None,
    extraction_output: Optional[Path] = None,
    registry=None,
    publication: Optional[asyncio.Lock] = None,
) -> None:
    result = None
    if extract:
        await seed_semantic(store)
        result = await process_material(
            document,
            mode=extractor,
            provider=provider,
            registry=registry
            if registry is not None
            else await resolve(store.read_concepts()),
            context_reader=getattr(store, "read_related_chunks", None),
        )
    async with publication or _NoLock():
        if result is not None:
            await resolve(store.write_processed(document, result))
            # Old claims about the same concepts meet the new ones.
            await reconcile_quietly(store, [document.document_version_id])
        else:
            await resolve(store.write_document(document))
    if result is not None:
        if extraction_output:
            extraction_output.parent.mkdir(parents=True, exist_ok=True)
            extraction_output.write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
        logger.info(
            "document=%s extraction=%s assertions=%d run=%s",
            document.document_version_id,
            result.run.status,
            len(result.assertions),
            result.run.run_id,
        )
    else:
        logger.info(
            "document=%s stored without extraction",
            document.document_version_id,
        )


class _NoLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


def _ingest(
    document,
    extract: bool,
    extractor: str = "llm",
    extraction_output: Optional[Path] = None,
) -> None:
    provider = _provider(extract, extractor)

    async def run():
        async with _opened(_store()) as store:
            await resolve(store.ensure_schema())
            await _write_ingested_async(
                document,
                extract,
                store,
                extractor=extractor,
                provider=provider,
                extraction_output=extraction_output,
            )

    asyncio.run(run())


def _provider(extract: bool, extractor: str):
    if extract and extractor == "llm":
        from .llm.client import JsonLLM

        return JsonLLM.from_environment()
    return None


def _crawl_run(platform: str, query: str, filter: Optional[str] = None):
    bounds = {}
    for part in (filter or "").split(","):
        key, separator, value = part.partition(":")
        if separator and key in (
            "from_publication_date",
            "to_publication_date",
        ):
            bounds[key] = date.fromisoformat(value).isoformat()
    return {
        "crawl_id": f"crawl:{uuid4()}",
        "source_id": f"source:{platform}",
        "source_family": load_catalog("sources")["platforms"][platform][
            "source_family"
        ],
        "query": query,
        "filter": filter,
        "period_start": bounds.get("from_publication_date"),
        "period_end": bounds.get("to_publication_date"),
        "records_seen": 0,
        "records_ingested": 0,
        # Works that failed to parse or extract; the search still saw them.
        "failures": 0,
        # Search pages that failed: their results were never seen.
        "search_failures": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "status": "running",
        # Proven only by a search that saw all its results. A registry
        # sample never proves absence.
        "exhaustive": False,
    }


@asynccontextmanager
async def _crawl_audit(store, run, checkpoint):
    await resolve(store.write_crawl_run(dict(run)))
    try:
        yield
    except BaseException:
        run["status"] = "failed"
        raise
    else:
        run["status"] = "completed"
    finally:
        run["finished_at"] = datetime.now(timezone.utc).isoformat()
        if checkpoint.exists():
            run["checkpoint_json"] = checkpoint.read_text(encoding="utf-8")
        try:
            await resolve(store.write_crawl_run(dict(run)))
        except Exception:
            logger.exception(
                "Failed to finalize crawl audit %s", run["crawl_id"]
            )


async def _job_registry(store, extract: bool):
    """One in-memory registry for a whole crawl, read from the graph once."""
    if not extract:
        return None
    await seed_semantic(store)
    return ConceptRegistry(await resolve(store.read_concepts()))


async def _record_metrics(store, document) -> None:
    """Counters of a skipped version are a new observation, not lost."""
    recorder = getattr(store, "record_metrics", None)
    if recorder is None:
        return
    try:
        await resolve(recorder(document))
    except Exception as exc:
        if type(exc).__module__.startswith("neo4j"):
            raise
        logger.warning(
            "Metrics of %s not recorded: %s",
            document.document_version_id,
            type(exc).__name__,
        )


def _crawl_openalex(
    query: str,
    limit: int,
    per_page: int,
    checkpoint: Path,
    extract: bool,
    extractor: str = "llm",
    fulltext: bool = True,
    filter: Optional[str] = None,
    workers: int = 1,
) -> None:
    if limit <= 0 or not 1 <= per_page <= 100:
        raise ValueError("limit must be positive and per-page must be 1..100")
    if not query.strip():
        raise ValueError("OpenAlex query must be nonempty")
    query = query.strip()
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1..16")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint.exists()
        else {}
    )
    if state and (
        state.get("query") != query or state.get("filter") != filter
    ):
        raise RuntimeError(
            "Checkpoint belongs to another query: "
            f"{state.get('query')!r} filter={state.get('filter')!r}"
        )
    processed = int(state.get("processed", 0))
    failures = int(state.get("failures", 0))
    cursor = state.get("cursor", "*")
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        raise ValueError("Invalid OpenAlex checkpoint cursor")
    logger.info(
        "OpenAlex crawl query=%r filter=%r limit=%d workers=%d resumed_at=%d",
        query,
        filter,
        limit,
        workers,
        processed,
    )
    started = perf_counter()
    provider = _provider(extract, extractor)
    if fulltext:
        require_pdf_support()

    async def one(payload, store, registry, slots, publication) -> bool:
        async with slots:
            try:
                raw = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                document = _snapshot(parse_openalex(payload, raw=raw), raw)
                prior = (
                    await prior_inputs(store, document.document_version_id)
                    if extract
                    else []
                )
                # Without --fulltext the input is known now; with it, a PDF
                # read by no earlier run still has to be tried.
                if prior and not fulltext and covers(prior, document):
                    logger.info(
                        "work=%s already processed; skipped",
                        document.source.record_id,
                    )
                    await _record_metrics(store, document)
                    return True
                if fulltext:
                    await resolve(
                        attach_openalex_fulltext(
                            document,
                            payload,
                            known_sha256=known_fulltexts(prior),
                        )
                    )
                    logger.info(
                        "work=%s fulltext=%s chunks=%d",
                        document.source.record_id,
                        document.metadata["fulltext"]["status"],
                        len(document.chunks),
                    )
                    if prior and covers(prior, document):
                        logger.info(
                            "work=%s already processed with this input; "
                            "skipped",
                            document.source.record_id,
                        )
                        await _record_metrics(store, document)
                        return True
                await _write_ingested_async(
                    document,
                    extract,
                    store,
                    extractor,
                    provider,
                    registry=registry,
                    publication=publication,
                )
                return True
            except Exception as exc:
                if type(exc).__module__.startswith("neo4j"):
                    raise  # the graph is unavailable: stop, keep checkpoint
                # One bad work (dead PDF link, provider error) must not stop
                # a crawl of thousands.
                logger.warning(
                    "work=%s failed: %s: %s",
                    payload.get("id"),
                    type(exc).__name__,
                    exc,
                )
                logger.debug("Work traceback", exc_info=True)
                return False

    async def crawl():
        nonlocal processed, failures, cursor
        from .llm.client import document_workers

        # One worker per LLM key: more would only queue for a busy key.
        slots = asyncio.Semaphore(document_workers(workers, provider))
        publication = asyncio.Lock()
        seen_cursors = set()
        crawl_run = _crawl_run("openalex", query, filter)
        # Number of works matching the search, as reported by OpenAlex.
        total = None
        async with (
            _opened(_store()) as store,
            _crawl_audit(store, crawl_run, checkpoint),
        ):
            await resolve(store.ensure_schema())
            registry = await _job_registry(store, extract)
            while processed < limit and cursor:
                if cursor in seen_cursors:
                    raise ValueError("Repeated OpenAlex cursor")
                seen_cursors.add(cursor)
                try:
                    page = await resolve(
                        fetch_openalex_page(
                            query,
                            cursor,
                            min(per_page, limit - processed),
                            os.getenv("OPENALEX_MAILTO"),
                            filter,
                        )
                    )
                except Exception:
                    crawl_run["search_failures"] += 1
                    raise
                if not isinstance(page, dict):
                    raise ValueError("Invalid OpenAlex page")
                works, meta = page.get("results"), page.get("meta", {})
                if isinstance(meta, dict) and isinstance(
                    meta.get("count"), int
                ):
                    total = meta["count"]
                if (
                    not isinstance(works, list)
                    or not isinstance(meta, dict)
                    or any(not isinstance(work, dict) for work in works)
                ):
                    raise ValueError("Invalid OpenAlex results")
                if works and "next_cursor" not in meta:
                    raise ValueError(
                        "OpenAlex response is missing next_cursor"
                    )
                next_cursor = meta.get("next_cursor")
                if next_cursor is not None and (
                    not isinstance(next_cursor, str) or not next_cursor
                ):
                    raise ValueError("Invalid OpenAlex cursor")
                if works and next_cursor in seen_cursors:
                    raise ValueError("Repeated OpenAlex cursor")
                if not works:
                    logger.info("OpenAlex returned no more works")
                    cursor = None
                    checkpoint.write_text(
                        json.dumps(
                            {
                                "query": query,
                                "filter": filter,
                                "processed": processed,
                                "failures": failures,
                                "cursor": cursor,
                            }
                        ),
                        encoding="utf-8",
                    )
                    break
                outcomes = await asyncio.gather(
                    *(
                        one(payload, store, registry, slots, publication)
                        for payload in works
                    )
                )
                processed += len(works)
                failures += outcomes.count(False)
                crawl_run["records_seen"] += len(works)
                crawl_run["records_ingested"] += outcomes.count(True)
                crawl_run["failures"] += outcomes.count(False)
                cursor = next_cursor
                checkpoint.write_text(
                    json.dumps(
                        {
                            "query": query,
                            "filter": filter,
                            "processed": processed,
                            "failures": failures,
                            "cursor": cursor,
                        }
                    ),
                    encoding="utf-8",
                )
                crawl_run["checkpoint_json"] = checkpoint.read_text(
                    encoding="utf-8"
                )
                await resolve(store.write_crawl_run(dict(crawl_run)))
                elapsed = perf_counter() - started
                logger.info(
                    "processed=%d/%d failures=%d elapsed=%.1fs "
                    "rate=%.2f works/s",
                    processed,
                    limit,
                    failures,
                    elapsed,
                    processed / elapsed,
                )
            # Seen by this run alone: a resumed crawl proves nothing about
            # the part an earlier run saw.
            crawl_run["exhaustive"] = (
                total is not None
                and crawl_run["records_seen"] >= total
                and crawl_run["search_failures"] == 0
            )

    asyncio.run(crawl())


def _crawl_economic(
    source: str,
    query: str,
    limit: int,
    checkpoint: Path,
    extract: bool,
    extractor: str = "llm",
    workers: int = 1,
) -> None:
    """Grants or vacancies of the economic layer (ingest.economic)."""
    from .ingest.economic import DISCOVERERS, PARSERS, hydrate_hh

    if source not in DISCOVERERS:
        raise ValueError(f"Unknown economic source {source!r}")
    query = query.strip()
    if limit <= 0 or not query:
        raise ValueError("limit must be positive and query nonempty")
    if not 1 <= workers <= 16:
        raise ValueError("workers must be 1..16")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint.exists()
        else {}
    )
    if state and (state.get("source"), state.get("query")) != (source, query):
        raise RuntimeError(
            "Checkpoint belongs to another crawl: "
            f"{state.get('source')!r} {state.get('query')!r}"
        )
    processed = int(state.get("processed", 0))
    cursor = state.get("cursor")
    if state and cursor is None:
        logger.info("%s crawl %r already finished", source, query)
        return
    provider = _provider(extract, extractor)

    def save() -> None:
        checkpoint.write_text(
            json.dumps(
                {
                    "source": source,
                    "query": query,
                    "processed": processed,
                    "cursor": cursor,
                }
            ),
            encoding="utf-8",
        )

    async def one(item, store, registry, slots, publication) -> bool:
        async with slots:
            try:
                payload = item["payload"]
                if source == "hh":
                    payload = await hydrate_hh(payload)
                raw = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                document = _snapshot(PARSERS[source](payload, raw=raw), raw)
                prior = (
                    await prior_inputs(store, document.document_version_id)
                    if extract
                    else []
                )
                if prior and covers(prior, document):
                    logger.info(
                        "%s=%s already processed; skipped",
                        source,
                        document.source.record_id,
                    )
                    return True
                await _write_ingested_async(
                    document,
                    extract,
                    store,
                    extractor,
                    provider,
                    registry=registry,
                    publication=publication,
                )
                return True
            except Exception as exc:
                if type(exc).__module__.startswith("neo4j"):
                    raise
                logger.warning(
                    "%s=%s failed: %s: %s",
                    source,
                    item.get("source_id"),
                    type(exc).__name__,
                    exc,
                )
                logger.debug("Record traceback", exc_info=True)
                return False

    async def crawl():
        nonlocal processed, cursor
        from .llm.client import document_workers

        # One worker per LLM key: more would only queue for a busy key.
        slots = asyncio.Semaphore(document_workers(workers, provider))
        publication = asyncio.Lock()
        crawl_run = _crawl_run(source, query)
        complete = False
        async with (
            _opened(_store()) as store,
            _crawl_audit(store, crawl_run, checkpoint),
        ):
            await resolve(store.ensure_schema())
            registry = await _job_registry(store, extract)
            while processed < limit:
                try:
                    page = await DISCOVERERS[source](query, cursor)
                except Exception:
                    crawl_run["search_failures"] += 1
                    raise
                for limitation in page["limitations"]:
                    logger.warning("%s: %s", source, limitation["message"])
                items = page["items"][: limit - processed]
                outcomes = await asyncio.gather(
                    *(
                        one(item, store, registry, slots, publication)
                        for item in items
                    )
                )
                processed += len(items)
                crawl_run["records_seen"] += len(items)
                crawl_run["records_ingested"] += outcomes.count(True)
                crawl_run["failures"] += outcomes.count(False)
                cursor = page["next_cursor"]
                complete = page["complete"] and cursor is None
                save()
                crawl_run["checkpoint_json"] = checkpoint.read_text(
                    encoding="utf-8"
                )
                await resolve(store.write_crawl_run(dict(crawl_run)))
                logger.info(
                    "%s processed=%d/%d total=%s",
                    source,
                    processed,
                    limit,
                    page["total"],
                )
                if cursor is None:
                    break
            crawl_run["exhaustive"] = (
                complete
                and not state
                and crawl_run["search_failures"] == 0
            )

    asyncio.run(crawl())


def _env_workers(default: int) -> int:
    """LCTREND_WORKERS, or the default when it is empty or not a number:
    a typo in .env must not break every command, --help included (H-8)."""
    value = os.getenv("LCTREND_WORKERS", "").strip()
    try:
        return int(value) if value else int(default)
    except ValueError:
        logger.warning(
            "LCTREND_WORKERS=%r is not a number; using %s", value, default
        )
        return int(default)


def _uniform_sample(
    names: List[str], limit: int, phase: float = 0.0
) -> List[str]:
    if limit <= 0 or not 0 <= phase < 1:
        raise ValueError("limit must be positive and phase must be in [0, 1)")
    if limit > len(names):
        raise RuntimeError(f"PyPI has only {len(names)} projects")
    return [
        names[min(len(names) - 1, int((index + phase) * len(names) / limit))]
        for index in range(limit)
    ]


def _crawl_pypi(
    limit: int,
    checkpoint: Path,
    extract: bool,
    sample_phase: float = 0.0,
    requested_packages: Optional[List[str]] = None,
    extractor: str = "llm",
) -> None:
    if limit <= 0:
        raise ValueError("limit must be positive")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(checkpoint.read_text(encoding="utf-8"))
        if checkpoint.exists()
        else {}
    )
    if (
        requested_packages
        and state
        and state.get("packages") != requested_packages
    ):
        raise ValueError("Checkpoint belongs to a different package list")
    packages = (
        state.get("packages")
        or requested_packages
        or _uniform_sample(
            asyncio.run(resolve(fetch_pypi_projects())),
            limit * 2,
            sample_phase,
        )
    )
    processed = int(state.get("processed", 0))
    successful = int(state.get("successful", 0))
    failures = int(state.get("failures", 0))
    logger.info(
        "PyPI crawl limit=%d candidates=%d resumed_at=%d",
        limit,
        len(packages),
        processed,
    )
    started = perf_counter()
    provider = _provider(extract, extractor)

    async def crawl():
        nonlocal processed, successful, failures
        crawl_run = _crawl_run(
            "pypi",
            json.dumps(requested_packages or {"sample_phase": sample_phase}),
        )
        async with (
            _opened(_store()) as store,
            _crawl_audit(store, crawl_run, checkpoint),
        ):
            await resolve(store.ensure_schema())
            registry = await _job_registry(store, extract)
            for package in packages[processed:]:
                if successful >= limit:
                    break
                crawl_run["records_seen"] += 1
                try:
                    payload = await resolve(fetch_pypi(package))
                    raw = json.dumps(
                        payload, ensure_ascii=False, sort_keys=True
                    ).encode("utf-8")
                    document = _snapshot(parse_pypi(payload, raw=raw), raw)
                    await _write_ingested_async(
                        document,
                        extract,
                        store,
                        extractor,
                        provider,
                        registry=registry,
                    )
                    successful += 1
                    crawl_run["records_ingested"] += 1
                except Exception as exc:
                    failures += 1
                    crawl_run["failures"] += 1
                    logger.warning("skipped=%s error=%s", package, exc)
                    logger.debug("Package %s failed", package, exc_info=True)
                processed += 1
                if (
                    processed % 25 == 0
                    or successful == limit
                    or processed == len(packages)
                ):
                    checkpoint.write_text(
                        json.dumps(
                            {
                                "packages": packages,
                                "processed": processed,
                                "successful": successful,
                                "failures": failures,
                            }
                        ),
                        encoding="utf-8",
                    )
                    crawl_run["checkpoint_json"] = checkpoint.read_text(
                        encoding="utf-8"
                    )
                    await resolve(store.write_crawl_run(dict(crawl_run)))
                    elapsed = perf_counter() - started
                    logger.info(
                        "successful=%d/%d processed=%d failures=%d "
                        "elapsed=%.1fs rate=%.2f packages/s",
                        successful,
                        limit,
                        processed,
                        failures,
                        elapsed,
                        processed / elapsed,
                    )

    asyncio.run(crawl())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    load_environment()
    log_path = setup_logging()
    from .core.catalog_validation import validate_catalogs

    validate_catalogs()
    settings = load_catalog("runtime")
    default_extractor = "llm"
    parser = argparse.ArgumentParser(prog="lctrend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_command = subparsers.add_parser(
        "parse", help="Parse a saved source response"
    )
    parse_command.add_argument("kind", choices=[*PARSERS, "epo", "file"])
    parse_command.add_argument("input", type=Path)
    parse_command.add_argument("--output", type=Path)

    fetch_command = subparsers.add_parser(
        "fetch", help="Fetch and parse a public API record"
    )
    fetch_command.add_argument("kind", choices=list(PARSERS))
    fetch_command.add_argument("identifier")
    fetch_command.add_argument("--output", type=Path)
    fetch_command.add_argument("--ingest", action="store_true")
    fetch_command.set_defaults(extract=True)
    fetch_command.add_argument(
        "--extract",
        dest="extract",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    fetch_command.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        help="Skip extraction during ingest",
    )
    fetch_command.add_argument(
        "--extractor", choices=EXTRACTORS, default=default_extractor
    )
    fetch_command.add_argument("--extraction-output", type=Path)
    fetch_command.add_argument(
        "--no-fulltext",
        dest="fulltext",
        action="store_false",
        help=(
            "OpenAlex: keep the abstract only, "
            "do not download the open-access PDF"
        ),
    )

    ingest_command = subparsers.add_parser(
        "ingest", help="Parse a saved record into Neo4j"
    )
    ingest_command.add_argument("kind", choices=[*PARSERS, "epo", "file"])
    ingest_command.add_argument("input", type=Path)
    ingest_command.set_defaults(extract=True)
    ingest_command.add_argument(
        "--extract",
        dest="extract",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ingest_command.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        help="Skip extraction during ingest",
    )
    ingest_command.add_argument(
        "--extractor", choices=EXTRACTORS, default=default_extractor
    )
    ingest_command.add_argument("--extraction-output", type=Path)

    crawl_command = subparsers.add_parser(
        "crawl-openalex", help="Crawl OpenAlex search results into Neo4j"
    )
    crawl_command.add_argument("query")
    crawl_command.add_argument(
        "--limit", type=int, default=settings["openalex"]["limit"]
    )
    crawl_command.add_argument(
        "--per-page",
        type=int,
        default=settings["openalex"]["per_page"],
        help="OpenAlex works per request, 1..100",
    )
    crawl_command.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(settings["openalex"]["checkpoint"]),
    )
    crawl_command.add_argument(
        "--filter",
        help=(
            'OpenAlex filter, e.g. "is_oa:true,has_abstract:true,'
            'type:article,from_publication_date:2024-01-01"'
        ),
    )
    crawl_command.add_argument(
        "--no-fulltext",
        dest="fulltext",
        action="store_false",
        help="Keep abstracts only, do not download open-access PDFs",
    )
    crawl_command.set_defaults(extract=True)
    crawl_command.add_argument(
        "--no-extract", dest="extract", action="store_false"
    )
    crawl_command.add_argument(
        "--extractor", choices=EXTRACTORS, default=default_extractor
    )
    crawl_command.add_argument(
        "--workers",
        type=int,
        default=_env_workers(settings.get("ingestion", {}).get("workers", 1)),
        help="Works processed concurrently (LLM, PDF, Neo4j), 1..16",
    )

    economic_command = subparsers.add_parser(
        "crawl-economic",
        help="Crawl grants (nih, nsf) or vacancies (trudvsem, hh) of the "
        "economic layer into Neo4j",
    )
    economic_command.add_argument(
        "source", choices=("nih", "nsf", "trudvsem", "hh")
    )
    economic_command.add_argument(
        "query",
        help="A phrase; vacancies (trudvsem, hh) are searched in Russian",
    )
    economic_command.add_argument("--limit", type=int, default=200)
    economic_command.add_argument(
        "--checkpoint",
        type=Path,
        help="Resume state (default: one file per source and query in "
        "artifacts/checkpoints)",
    )
    economic_command.set_defaults(extract=True)
    economic_command.add_argument(
        "--no-extract", dest="extract", action="store_false"
    )
    economic_command.add_argument(
        "--extractor", choices=EXTRACTORS, default=default_extractor
    )
    economic_command.add_argument(
        "--workers",
        type=int,
        default=_env_workers(settings.get("ingestion", {}).get("workers", 1)),
        help="Records processed concurrently, 1..16",
    )

    pypi_crawl_command = subparsers.add_parser(
        "crawl-pypi", help="Crawl a uniform PyPI sample into Neo4j"
    )
    pypi_crawl_command.add_argument(
        "--limit", type=int, default=settings["pypi"]["limit"]
    )
    pypi_crawl_command.add_argument(
        "--checkpoint", type=Path, default=Path(settings["pypi"]["checkpoint"])
    )
    pypi_crawl_command.add_argument(
        "--sample-phase",
        type=float,
        default=settings["pypi"]["sample_phase"],
        help=argparse.SUPPRESS,
    )
    pypi_crawl_command.add_argument(
        "--packages", nargs="+", help="Specific PyPI packages to ingest"
    )
    pypi_crawl_command.set_defaults(extract=True)
    pypi_crawl_command.add_argument(
        "--no-extract", dest="extract", action="store_false"
    )
    pypi_crawl_command.add_argument(
        "--extractor", choices=EXTRACTORS, default=default_extractor
    )

    dataset_settings = load_catalog("dataset")
    training_command = subparsers.add_parser(
        "build-training-set",
        help="Export technology snapshots and future realization labels",
    )
    training_command.add_argument(
        "--output", type=Path, default=Path(settings["training_output"])
    )
    training_command.add_argument(
        "--start-year",
        type=int,
        default=dataset_settings["snapshots"]["start_year"],
    )
    training_command.add_argument(
        "--horizon-years",
        type=int,
        default=dataset_settings["horizon_years"],
    )
    training_command.add_argument(
        "--min-documents",
        type=int,
        default=dataset_settings["min_documents"],
    )
    training_command.add_argument(
        "--end-date",
        type=date.fromisoformat,
        help="Last snapshot date (YYYY-MM-DD; defaults to corpus end)",
    )
    training_command.add_argument(
        "--subgraphs-output",
        type=Path,
        help="Also export bounded point-in-time subgraphs as JSONL",
    )
    training_command.add_argument(
        "--model-grid",
        action="store_true",
        help="Use annual/half-yearly/quarterly historical model snapshots",
    )
    training_command.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Skip semantic, taxonomy and graph novelty features",
    )
    training_command.add_argument(
        "--as-known",
        action="store_true",
        help="Strict mode: content also waits for its collection and "
        "extraction (what this system knew at T), not only publication",
    )
    features_command = subparsers.add_parser(
        "export-features", help="Export snapshot graph features"
    )
    features_command.add_argument("--snapshot", required=True)
    features_command.add_argument(
        "--output", type=Path, default=Path(settings["features_output"])
    )
    features_command.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Skip semantic, taxonomy and graph novelty features",
    )
    features_command.add_argument(
        "--as-known",
        action="store_true",
        help="Strict mode: content also waits for its collection and "
        "extraction (what this system knew at T), not only publication",
    )
    ranking_settings = load_catalog("ranking")
    backtest_command = subparsers.add_parser(
        "backtest-top15",
        help="TOP-K at a date from data up to it, then the share that grew "
        "in the horizon against random samples (precision@K)",
    )
    backtest_command.add_argument(
        "--snapshot", type=date.fromisoformat, required=True
    )
    backtest_command.add_argument(
        "--horizon-years",
        type=int,
        default=ranking_settings["backtest"]["horizon_years"],
    )
    backtest_command.add_argument(
        "--top-k", type=int, default=ranking_settings["top_k"]
    )
    backtest_command.add_argument(
        "--trials",
        type=int,
        default=ranking_settings["backtest"]["random_trials"],
        help="Random samples of the candidate pool",
    )
    backtest_command.add_argument(
        "--seed", type=int, default=ranking_settings["backtest"]["seed"]
    )
    backtest_command.add_argument(
        "--output", type=Path, help="Also write the full result as JSON"
    )
    backtest_command.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Skip semantic, taxonomy and graph novelty features",
    )
    backtest_command.add_argument(
        "--as-known",
        action="store_true",
        help="Strict mode: content also waits for its collection and "
        "extraction (what this system knew at T), not only publication",
    )
    signals_command = subparsers.add_parser(
        "signals-report",
        help="Weak-signal table at a date: niche, area, companies, why, "
        "stage, trend, score and sources (XLSX/CSV/JSON)",
    )
    signals_command.add_argument(
        "--snapshot",
        type=date.fromisoformat,
        default=date.today(),
        help="Snapshot date T (default: today)",
    )
    signals_command.add_argument(
        "--output",
        type=Path,
        action="append",
        required=True,
        help="Report file: .xlsx, .csv or .json; repeat for several",
    )
    signals_command.add_argument(
        "--top-k",
        type=int,
        default=ranking_settings["signals"]["top_k"],
        help="Cards in the table",
    )
    signals_command.add_argument(
        "--pool",
        type=int,
        default=ranking_settings["signals"]["pool"],
        help="Ranked candidates read before clustering",
    )
    signals_command.add_argument(
        "--query",
        help="Narrow to domains or technology names, like /api/search",
    )
    signals_command.add_argument(
        "--no-llm",
        action="store_true",
        help="Cards from the dossier only, without the language model",
    )
    signals_command.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Skip semantic, taxonomy and graph novelty features",
    )
    signals_command.add_argument(
        "--as-known",
        action="store_true",
        help="Strict mode: content also waits for its collection and "
        "extraction (what this system knew at T), not only publication",
    )
    taxonomy_command = subparsers.add_parser(
        "build-taxonomy",
        help="Build the technology taxonomy of a snapshot into Neo4j",
    )
    taxonomy_command.add_argument("--snapshot", required=True)
    taxonomy_command.add_argument(
        "--output",
        type=Path,
        help="Also write the tree as JSON (default artifacts/taxonomy/"
        "<snapshot>.json)",
    )

    prune_command = subparsers.add_parser(
        "prune-chunks",
        help="Remove chunk nodes nothing stands on (no mention, quote, "
        "evidence or vector); a dry run without --apply",
    )
    prune_command.add_argument("--apply", action="store_true")
    subparsers.add_parser(
        "reconcile-claims",
        help="Rebuild CORROBORATES / CONTRADICTS / SHARES_CONTEXT_WITH: "
        "claims of the whole graph compared by slot",
    )
    similar_command = subparsers.add_parser(
        "similar-rebuild",
        help="Rebuild SIMILAR_TO: mutual nearest concepts by their stored "
        "vectors (computed edges without evidence)",
    )
    similar_command.add_argument(
        "--model",
        help="Embedding model of the vectors (default: the semantic layer's)",
    )
    similar_command.add_argument("--k", type=int, help="Neighbours per node")
    similar_command.add_argument("--min-cosine", type=float)
    similar_command.add_argument(
        "--dry-run", action="store_true", help="Count the edges, write none"
    )

    merge_command = subparsers.add_parser(
        "merge-concepts",
        help="Merge a duplicate concept into another of its kind family",
    )
    merge_command.add_argument("source", help="concept_id to merge away")
    merge_command.add_argument("target", help="concept_id that remains")
    merge_command.add_argument("--reason", help="Why they are one concept")

    migrate_command = subparsers.add_parser(
        "migrate-concept-keys",
        help="Recompute identity keys v2 of stored concepts and merge the "
        "duplicates they reveal (a dry run without --apply)",
    )
    migrate_command.add_argument(
        "--apply", action="store_true", help="Write keys and run the merges"
    )

    kind_command = subparsers.add_parser(
        "set-concept-kind",
        help="Review a concept's kind within its family, e.g. a compound "
        "extracted as a Technology is a Material",
    )
    kind_command.add_argument("concept_id")
    kind_command.add_argument(
        "kind", choices=["Technology", "Method", "Material"]
    )

    review_command = subparsers.add_parser(
        "review-duplicates",
        help="Review merge candidates with their graph context; merge the "
        "aliases sources declared (a dry run without --apply)",
    )
    review_command.add_argument(
        "--apply", action="store_true", help="Run the planned merges"
    )
    review_command.add_argument(
        "--merge-above",
        type=float,
        help="Also merge semantic candidates with this cross-encoder score "
        "or more (none by default)",
    )
    review_command.add_argument(
        "--limit", type=int, help="Review at most this many pairs"
    )

    works_command = subparsers.add_parser(
        "link-works",
        help="Group stored documents into works: one paper from OpenAlex, "
        "a PDF and arXiv, one patent as A1 and B1 (a dry run without "
        "--apply)",
    )
    works_command.add_argument(
        "--apply", action="store_true", help="Write the works"
    )

    normalize_command = subparsers.add_parser(
        "normalize-graph",
        help="Full country names, one node per company, organization types "
        "from the catalog (a dry run without --apply)",
    )
    normalize_command.add_argument(
        "--apply", action="store_true", help="Write the changes"
    )

    embed_command = subparsers.add_parser(
        "embed-concepts",
        help="Compute missing label vectors of stored concepts (novelty and "
        "taxonomy read them)",
    )
    embed_command.add_argument(
        "--force",
        action="store_true",
        help="Recompute every vector, not only missing or other-model ones",
    )

    subparsers.add_parser("init-graph", help="Create Neo4j constraints")
    args = parser.parse_args()
    logger.debug("Command %s started, log file: %s", args.command, log_path)
    try:
        _run(args)
    except Exception as exc:
        logger.exception(
            "Command %s failed: %s: %s (details: %s)",
            args.command,
            type(exc).__name__,
            exc,
            log_path,
        )
        raise SystemExit(1) from None
    logger.debug("Command %s finished", args.command)


async def _graph(action):
    # ``_store`` is looked up per call, so tests can replace it.
    return await with_graph(action, _store)


async def _ensure_schema(store):
    await resolve(store.ensure_schema())


async def _taxonomy_data(store, snapshot=None):
    kinds = load_catalog("taxonomy")["kinds"]
    return await resolve(store.read_taxonomy_input(kinds, snapshot=snapshot))


def _taxonomy(snapshot: str, data) -> Any:
    rows, parents = data
    return build_taxonomy(
        [TaxonomyConcept.from_row(row) for row in rows], snapshot, parents
    )


def _taxonomy_tree(taxonomy) -> Dict[str, Any]:
    def node(node_id: str) -> Dict[str, Any]:
        item = taxonomy.nodes[node_id]
        return {
            "node_id": item.node_id,
            "label": item.label,
            "level": item.level,
            "size": len(item.subtree_ids),
            "new_share": round(item.new_share, 4),
            "documents_last_year": item.documents_last_year,
            "documents_previous_year": item.documents_previous_year,
            "concepts": [
                {
                    "concept_id": cid,
                    "label": taxonomy.concepts[cid].label,
                    "general_term": cid in taxonomy.general_terms,
                }
                for cid in item.concept_ids
            ],
            "children": [node(child) for child in item.children],
        }

    return {
        "taxonomy_version": taxonomy.version,
        "snapshot": taxonomy.snapshot.isoformat(),
        "root": node(taxonomy.root().node_id),
    }


def _run(args: argparse.Namespace) -> None:
    if args.command == "init-graph":
        asyncio.run(_graph(_ensure_schema))
        return

    if args.command == "migrate-concept-keys":
        from .graph.migration import apply_key_migration

        plan = asyncio.run(
            _graph(lambda store: apply_key_migration(store, args.apply))
        )
        print(
            json.dumps(plan.summary(args.apply), ensure_ascii=False, indent=2)
        )
        return

    if args.command == "set-concept-kind":
        summary = asyncio.run(
            _graph(
                lambda store: resolve(
                    store.set_concept_kind(args.concept_id, args.kind)
                )
            )
        )
        print(json.dumps(summary, ensure_ascii=False))
        return

    if args.command == "review-duplicates":
        from .graph.review import review_duplicates

        plan = asyncio.run(
            _graph(
                lambda store: review_duplicates(
                    store, args.apply, args.merge_above, args.limit
                )
            )
        )
        print(
            json.dumps(plan.summary(args.apply), ensure_ascii=False, indent=2)
        )
        return

    if args.command == "link-works":
        from .graph.works import link_stored_works

        summary = asyncio.run(
            _graph(lambda store: link_stored_works(store, args.apply))
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "normalize-graph":
        from .graph.normalize import normalize_graph

        plan = asyncio.run(
            _graph(lambda store: normalize_graph(store, args.apply))
        )
        print(
            json.dumps(plan.summary(args.apply), ensure_ascii=False, indent=2)
        )
        return

    if args.command == "embed-concepts":
        summary = asyncio.run(
            _graph(lambda store: _embed_concepts(store, args.force))
        )
        print(json.dumps(summary, ensure_ascii=False))
        return

    if args.command == "merge-concepts":
        summary = asyncio.run(
            _graph(
                lambda store: resolve(
                    store.merge_concepts(
                        args.source, args.target, reason=args.reason
                    )
                )
            )
        )
        print(json.dumps(summary, ensure_ascii=False))
        return

    if args.command == "crawl-openalex":
        _crawl_openalex(
            args.query,
            args.limit,
            args.per_page,
            args.checkpoint,
            args.extract,
            args.extractor,
            fulltext=args.fulltext,
            filter=args.filter,
            workers=args.workers,
        )
        return

    if args.command == "crawl-economic":
        _crawl_economic(
            args.source,
            args.query,
            args.limit,
            args.checkpoint
            or Path("artifacts/checkpoints")
            / (
                f"{args.source}-"
                + stable_id("query", args.query.strip().casefold())[-12:]
                + ".json"
            ),
            args.extract,
            args.extractor,
            workers=args.workers,
        )
        return

    if args.command == "crawl-pypi":
        _crawl_pypi(
            len(args.packages) if args.packages else args.limit,
            args.checkpoint,
            args.extract,
            args.sample_phase,
            args.packages,
            args.extractor,
        )
        return

    if args.command == "build-training-set":
        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        rows = build_dataset_rows(
            corpus,
            start_year=args.start_year,
            horizon_years=args.horizon_years,
            min_documents=args.min_documents,
            end_date=args.end_date,
            include_novelty=not args.no_taxonomy,
            model_grid=args.model_grid,
        )
        logger.info(
            "rows=%d output=%s",
            write_dataset_rows(args.output, rows, corpus),
            args.output,
        )
        if args.subgraphs_output:
            config = load_catalog("dataset")["subgraph"]
            feature_fields = dataset_feature_fields()
            samples = (
                sample_subgraph(
                    corpus.view(date.fromisoformat(row["snapshot_date"])),
                    row["technology_id"],
                    label=row["label_realized"],
                    split=row["split"],
                    config=config,
                    features={name: row.get(name) for name in feature_fields},
                )
                for row in rows
            )
            logger.info(
                "subgraphs=%d output=%s",
                write_subgraph_rows(args.subgraphs_output, samples),
                args.subgraphs_output,
            )
        return

    if args.command == "backtest-top15":
        from .ranking.scoring import backtest

        if min(args.horizon_years, args.top_k, args.trials) < 1:
            raise ValueError("horizon, top-k and trials must be positive")
        config = copy.deepcopy(load_catalog("ranking"))
        config["top_k"] = args.top_k
        config["include_novelty"] = not args.no_taxonomy
        config["backtest"].update(
            horizon_years=args.horizon_years,
            random_trials=args.trials,
            seed=args.seed,
        )
        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        result = backtest(corpus, args.snapshot, config)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        random_mean = result["random"]["mean_precision"]
        print(
            f"{result['snapshot']} -> {result['horizon_end']}: "
            f"precision@{result['k']}={result['precision_at_k']} "
            f"random={random_mean} p={result['random']['p_value']} "
            f"candidates={result['candidates']}"
        )
        for warning in result["warnings"]:
            logger.warning("Backtest: %s", warning)
        return

    if args.command == "signals-report":
        from .ranking.export import write_report
        from .ranking.signals import build_cards

        if min(args.top_k, args.pool) < 1:
            raise ValueError("top-k and pool must be positive")
        config = copy.deepcopy(load_catalog("ranking"))
        config["include_novelty"] = not args.no_taxonomy
        config["signals"]["pool"] = args.pool
        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        provider = None
        if not args.no_llm:
            from .llm.client import JsonLLM

            provider = JsonLLM.from_environment()
        result = asyncio.run(
            build_cards(
                corpus,
                args.snapshot,
                provider,
                config,
                query=args.query,
                top_k=args.top_k,
            )
        )
        for path in args.output:
            write_report(result, path)
            logger.info("Signals report: %s", path)
        stats = result["stats"]
        print(
            f"{result['snapshot']}: cards={stats['cards']} "
            f"clusters={stats['clusters']} candidates={stats['candidates']} "
            f"rejected_by_model={stats['rejected_by_model']} "
            f"without_model={stats['without_model']}"
        )
        return

    if args.command == "prune-chunks":
        summary = asyncio.run(
            _graph(lambda store: store.prune_text_chunks(apply=args.apply))
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "reconcile-claims":
        from .linking.reconcile import reconcile

        summary = asyncio.run(_graph(reconcile))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "similar-rebuild":
        from .extraction.processing import _semantic_deduplicator
        from .linking.similar import rebuild

        model = args.model or _semantic_deduplicator().embedding_model_name
        summary = asyncio.run(
            _graph(
                lambda store: rebuild(
                    store,
                    model,
                    k=args.k,
                    min_cosine=args.min_cosine,
                    dry_run=args.dry_run,
                )
            )
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "build-taxonomy":
        taxonomy = _taxonomy(
            args.snapshot,
            asyncio.run(
                _graph(lambda store: _taxonomy_data(store, args.snapshot))
            ),
        )

        async def write(store):
            await resolve(store.ensure_schema())
            await resolve(store.write_taxonomy(taxonomy))

        asyncio.run(_graph(write))
        output = args.output or Path(
            "artifacts", "taxonomy", f"{taxonomy.snapshot.isoformat()}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(_taxonomy_tree(taxonomy), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "taxonomy=%s nodes=%d concepts=%d output=%s",
            taxonomy.version,
            len(taxonomy.nodes),
            len(taxonomy.placement),
            output,
        )
        return

    if args.command == "export-features":
        corpus = asyncio.run(
            _graph(lambda store: _temporal_data(store, args.as_known))
        )
        rows = build_snapshot_rows(
            corpus,
            args.snapshot,
            include_novelty=not args.no_taxonomy,
        )
        logger.info(
            "rows=%d output=%s",
            write_snapshot_rows(args.output, rows, corpus),
            args.output,
        )
        return

    if args.command == "fetch":
        fetchers = {
            "openalex": lambda: fetch_openalex(
                args.identifier, os.getenv("OPENALEX_MAILTO")
            ),
            "github": lambda: fetch_github(
                args.identifier, os.getenv("GITHUB_TOKEN")
            ),
            "pypi": lambda: fetch_pypi(args.identifier),
        }
        payload: Dict[str, Any] = asyncio.run(resolve(fetchers[args.kind]()))
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        document = _snapshot(PARSERS[args.kind](payload, raw=raw), raw)
        if args.kind == "openalex" and args.fulltext:
            asyncio.run(resolve(attach_openalex_fulltext(document, payload)))
        if args.output:
            args.output.write_text(
                document.model_dump_json(indent=2), encoding="utf-8"
            )
        else:
            print(document.model_dump_json(indent=2))
        if args.ingest:
            _ingest(
                document,
                args.extract,
                args.extractor,
                args.extraction_output,
            )
        return

    document = _parse(args.kind, args.input)
    if args.command == "parse":
        if args.output:
            args.output.write_text(
                document.model_dump_json(indent=2), encoding="utf-8"
            )
        else:
            print(document.model_dump_json(indent=2))
    else:
        _ingest(
            document,
            args.extract,
            args.extractor,
            args.extraction_output,
        )
