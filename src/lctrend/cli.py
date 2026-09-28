from __future__ import annotations

import argparse
import asyncio
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
from .extraction.processing import process_material
from .extraction.resolver import ConceptRegistry
from .graph.store import GraphStore
from .graph.subgraphs import sample_subgraph, write_subgraph_rows
from .graph.temporal import TemporalCorpus
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
from .ingest.snapshots import persist_snapshot as _snapshot
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


def _store() -> GraphStore:
    load_environment()
    return GraphStore(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "change-me-now"),
    )


@asynccontextmanager
async def _opened(store):
    """Open a graph store; offline test doubles may be synchronous."""
    if hasattr(store, "__aenter__"):
        async with store as opened:
            yield opened
    else:
        with store as opened:
            yield opened


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


def _write_ingested(*args, **kwargs) -> None:
    asyncio.run(_write_ingested_async(*args, **kwargs))


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
            "from_publication_date", "to_publication_date"
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
        "failures": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "status": "running",
        # A limited topic search or registry sample cannot prove absence.
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
                lookup = getattr(store, "processed_versions", None)
                if extract and lookup is not None:
                    done = await resolve(
                        lookup([document.document_version_id])
                    )
                    if document.document_version_id in done:
                        logger.info(
                            "work=%s already processed; skipped",
                            document.source.record_id,
                        )
                        await _record_metrics(store, document)
                        return True
                if fulltext:
                    await resolve(attach_openalex_fulltext(document, payload))
                    logger.info(
                        "work=%s fulltext=%s chunks=%d",
                        document.source.record_id,
                        document.metadata["fulltext"]["status"],
                        len(document.chunks),
                    )
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
        slots, publication = asyncio.Semaphore(workers), asyncio.Lock()
        seen_cursors = set()
        crawl_run = _crawl_run("openalex", query, filter)
        async with _opened(_store()) as store, _crawl_audit(
            store, crawl_run, checkpoint
        ):
            await resolve(store.ensure_schema())
            registry = await _job_registry(store, extract)
            while processed < limit and cursor:
                if cursor in seen_cursors:
                    raise ValueError("Repeated OpenAlex cursor")
                seen_cursors.add(cursor)
                page = await resolve(
                    fetch_openalex_page(
                        query,
                        cursor,
                        min(per_page, limit - processed),
                        os.getenv("OPENALEX_MAILTO"),
                        filter,
                    )
                )
                if not isinstance(page, dict):
                    raise ValueError("Invalid OpenAlex page")
                works, meta = page.get("results"), page.get("meta", {})
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

    asyncio.run(crawl())


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
        async with _opened(_store()) as store, _crawl_audit(
            store, crawl_run, checkpoint
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
        default=int(
            os.getenv("LCTREND_WORKERS", "").strip()
            or settings.get("ingestion", {}).get("workers", 1)
        ),
        help="Works processed concurrently (LLM, PDF, Neo4j), 1..16",
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
        "--output", type=Path, default=Path(settings["training"]["output"])
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
    async with _opened(_store()) as store:
        return await action(store)


async def _ensure_schema(store):
    await resolve(store.ensure_schema())


async def _temporal_data(store, as_known=False):
    return TemporalCorpus(
        await resolve(store.read_temporal_data()), as_known=as_known
    )


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
