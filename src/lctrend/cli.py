from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from .core.config import load_catalog, load_environment
from .core.logging_config import setup_logging
from .extraction.processing import (
    _semantic_deduplicator,
    process_material,
)
from .graph.store import GraphStore
from .graph.training import (
    build_feature_rows,
    build_training_rows,
    write_feature_rows,
    write_training_rows,
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

logger = logging.getLogger(__name__)

EXTRACTORS = ["hybrid", "llm", "gliner"]
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
        document = parse_epo(raw.decode("utf-8"), path.resolve().as_uri())
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


def _load_ner_model(model_name: str):
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            'Install NER support first: pip install -e ".[ner]"'
        ) from exc

    return GLiNER.from_pretrained(model_name)


def _auxiliary_ner(extract: bool, extractor: str, model_name: str):
    """Hybrid mode keeps working as LLM-only when GLiNER cannot be loaded."""
    if not extract or extractor != "hybrid":
        return None
    try:
        return _load_ner_model(model_name)
    except Exception as exc:
        logger.warning(
            "Auxiliary NER disabled (%s: %s)", type(exc).__name__, exc
        )
        return None


def _write_ingested(
    document,
    extract: bool,
    model_name: str,
    store,
    model=None,
    semantic=None,
    extractor: str = "llm",
    provider=None,
    extraction_output: Optional[Path] = None,
) -> None:
    result = None
    if extract:
        result = process_material(
            document,
            mode=extractor,
            provider=provider,
            ner_model=(model or _load_ner_model(model_name))
            if extractor == "gliner"
            else model,
            model_name=model_name,
            registry=store.read_concepts(),
            semantic=(semantic or _semantic_deduplicator())
            if extractor == "gliner"
            else None,
        )
    if result is not None:
        store.write_processed(document, result)
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
        store.write_document(document)
        logger.info(
            "document=%s stored without extraction",
            document.document_version_id,
        )


def _ingest(
    document,
    extract: bool,
    model_name: str,
    extractor: str = "llm",
    extraction_output: Optional[Path] = None,
) -> None:
    provider = _provider(extract, extractor)
    model = _auxiliary_ner(extract, extractor, model_name)
    with _store() as store:
        store.ensure_schema()
        _write_ingested(
            document,
            extract,
            model_name,
            store,
            model,
            extractor=extractor,
            provider=provider,
            extraction_output=extraction_output,
        )


def _provider(extract: bool, extractor: str):
    if extract and extractor in ("llm", "hybrid"):
        from .llm.client import JsonLLM

        return JsonLLM.from_environment()
    return None


def _crawl_openalex(
    query: str,
    limit: int,
    per_page: int,
    checkpoint: Path,
    extract: bool,
    model_name: str,
    extractor: str = "llm",
    fulltext: bool = True,
    filter: Optional[str] = None,
) -> None:
    if limit <= 0 or not 1 <= per_page <= 200:
        raise ValueError("limit must be positive and per-page must be 1..200")
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
    cursor = state.get("cursor", "*")
    logger.info(
        "OpenAlex crawl query=%r filter=%r limit=%d resumed_at=%d",
        query,
        filter,
        limit,
        processed,
    )
    started = perf_counter()
    provider = _provider(extract, extractor)
    if fulltext:
        require_pdf_support()
    model = (
        _load_ner_model(model_name)
        if extract and extractor == "gliner"
        else _auxiliary_ner(extract, extractor, model_name)
    )
    semantic = (
        _semantic_deduplicator() if extract and extractor == "gliner" else None
    )
    with _store() as store:
        store.ensure_schema()
        while processed < limit and cursor:
            page = fetch_openalex_page(
                query,
                cursor,
                min(per_page, limit - processed),
                os.getenv("OPENALEX_MAILTO"),
                filter,
            )
            works = page.get("results", [])
            if not works:
                logger.info("OpenAlex returned no more works")
                break
            for payload in works:
                raw = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                document = _snapshot(parse_openalex(payload, raw=raw), raw)
                if fulltext:
                    attach_openalex_fulltext(document, payload)
                    logger.info(
                        "work=%s fulltext=%s chunks=%d",
                        document.source.record_id,
                        document.metadata["fulltext"]["status"],
                        len(document.chunks),
                    )
                _write_ingested(
                    document,
                    extract,
                    model_name,
                    store,
                    model,
                    semantic,
                    extractor,
                    provider,
                )
                processed += 1
            cursor = page.get("meta", {}).get("next_cursor")
            checkpoint.write_text(
                json.dumps(
                    {
                        "query": query,
                        "filter": filter,
                        "processed": processed,
                        "cursor": cursor,
                    }
                ),
                encoding="utf-8",
            )
            elapsed = perf_counter() - started
            logger.info(
                "processed=%d/%d elapsed=%.1fs rate=%.2f works/s",
                processed,
                limit,
                elapsed,
                processed / elapsed,
            )


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
    model_name: str,
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
        or _uniform_sample(fetch_pypi_projects(), limit * 2, sample_phase)
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
    model = (
        _load_ner_model(model_name)
        if extract and extractor == "gliner"
        else _auxiliary_ner(extract, extractor, model_name)
    )
    semantic = (
        _semantic_deduplicator() if extract and extractor == "gliner" else None
    )
    with _store() as store:
        store.ensure_schema()
        for package in packages[processed:]:
            if successful >= limit:
                break
            try:
                payload = fetch_pypi(package)
                raw = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")
                document = _snapshot(parse_pypi(payload, raw=raw), raw)
                _write_ingested(
                    document,
                    extract,
                    model_name,
                    store,
                    model,
                    semantic,
                    extractor,
                    provider,
                )
                successful += 1
            except Exception as exc:
                failures += 1
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    load_environment()
    log_path = setup_logging()
    settings = load_catalog("runtime")
    ner_model = os.getenv("GLINER_MODEL", "").strip() or settings["ner_model"]
    default_extractor = (
        os.getenv("LCTREND_EXTRACTOR", "").strip()
        or settings["default_extractor"]
    )
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
    fetch_command.add_argument("--ner-model", default=ner_model)

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
    ingest_command.add_argument("--ner-model", default=ner_model)

    crawl_command = subparsers.add_parser(
        "crawl-openalex", help="Crawl OpenAlex search results into Neo4j"
    )
    crawl_command.add_argument("query")
    crawl_command.add_argument(
        "--limit", type=int, default=settings["openalex"]["limit"]
    )
    crawl_command.add_argument(
        "--per-page", type=int, default=settings["openalex"]["per_page"]
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
    crawl_command.add_argument("--ner-model", default=ner_model)

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
    pypi_crawl_command.add_argument("--ner-model", default=ner_model)

    training_command = subparsers.add_parser(
        "build-training-set", help="Export temporal technology training rows"
    )
    training_command.add_argument(
        "--output", type=Path, default=Path(settings["training"]["output"])
    )
    training_command.add_argument(
        "--start-year", type=int, default=settings["training"]["start_year"]
    )
    training_command.add_argument(
        "--horizon-years",
        type=int,
        default=settings["training"]["horizon_years"],
    )
    training_command.add_argument(
        "--min-documents",
        type=int,
        default=settings["training"]["min_documents"],
    )
    training_command.add_argument(
        "--positive-future-documents",
        type=int,
        default=settings["training"]["positive_future_documents"],
    )
    training_command.add_argument(
        "--negative-future-documents",
        type=int,
        default=settings["training"]["negative_future_documents"],
    )
    features_command = subparsers.add_parser(
        "export-features", help="Export snapshot graph features"
    )
    features_command.add_argument("--snapshot", required=True)
    features_command.add_argument(
        "--output", type=Path, default=Path(settings["features_output"])
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


def _run(args: argparse.Namespace) -> None:
    if args.command == "init-graph":
        with _store() as store:
            store.ensure_schema()
        return

    if args.command == "crawl-openalex":
        _crawl_openalex(
            args.query,
            args.limit,
            args.per_page,
            args.checkpoint,
            args.extract,
            args.ner_model,
            args.extractor,
            fulltext=args.fulltext,
            filter=args.filter,
        )
        return

    if args.command == "crawl-pypi":
        _crawl_pypi(
            len(args.packages) if args.packages else args.limit,
            args.checkpoint,
            args.extract,
            args.ner_model,
            args.sample_phase,
            args.packages,
            args.extractor,
        )
        return

    if args.command == "build-training-set":
        with _store() as store:
            mentions, documents, tasks = store.read_training_data()
        rows = build_training_rows(
            mentions,
            documents,
            tasks,
            args.start_year,
            args.horizon_years,
            args.min_documents,
            args.positive_future_documents,
            args.negative_future_documents,
        )
        logger.info(
            "rows=%d output=%s",
            write_training_rows(args.output, rows),
            args.output,
        )
        return

    if args.command == "export-features":
        with _store() as store:
            mentions, documents, tasks = store.read_training_data()
        rows = build_feature_rows(mentions, documents, tasks, args.snapshot)
        logger.info(
            "rows=%d output=%s",
            write_feature_rows(args.output, rows),
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
        payload: Dict[str, Any] = fetchers[args.kind]()
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
        document = _snapshot(PARSERS[args.kind](payload, raw=raw), raw)
        if args.kind == "openalex" and args.fulltext:
            attach_openalex_fulltext(document, payload)
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
                args.ner_model,
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
            args.ner_model,
            args.extractor,
            args.extraction_output,
        )
