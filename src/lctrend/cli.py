from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

from .adapters import parse_epo, parse_github, parse_openalex, parse_pypi
from .connectors import (
    fetch_github,
    fetch_openalex,
    fetch_openalex_page,
    fetch_pypi,
    fetch_pypi_projects,
)
from .economics import extract_economic_evidence
from .graph import GraphStore
from .models import ExtractionResult, ProcessingRun, stable_id
from .ner import extract_mentions
from .resolver import SemanticDeduplicator, resolve_mentions


PARSERS = {
    "openalex": parse_openalex,
    "github": parse_github,
    "pypi": parse_pypi,
}


def _parse(kind: str, path: Path):
    raw = path.read_bytes()
    if kind == "epo":
        return parse_epo(raw.decode("utf-8"), path.resolve().as_uri())
    payload = json.loads(raw)
    return PARSERS[kind](payload, raw=raw)


def _store() -> GraphStore:
    return GraphStore(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "change-me-now"),
    )


def _load_ner_model(model_name: str):
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError('Install NER support first: pip install -e ".[ner]"') from exc

    return GLiNER.from_pretrained(model_name)


def _extract(document, model_name: str, registry, semantic: SemanticDeduplicator, model) -> ExtractionResult:
    mentions = extract_mentions(document, model)
    concepts, resolutions = resolve_mentions(mentions, registry, semantic)
    economic_evidence = extract_economic_evidence(
        document.chunks, mentions, concepts, resolutions
    )
    started_at = datetime.now(timezone.utc).isoformat()
    return ExtractionResult(
        document_version_id=document.document_version_id,
        run=ProcessingRun(
            run_id=stable_id("run", document.document_version_id, model_name, started_at),
            parser="gliner",
            model_revision=model_name,
            config_hash=stable_id("config", model_name, "threshold=0.5"),
            started_at=started_at,
        ),
        mentions=mentions,
        concepts=concepts,
        resolutions=resolutions,
        economic_evidence=economic_evidence,
    )


def _semantic_deduplicator() -> SemanticDeduplicator:
    return SemanticDeduplicator(
        cosine_threshold=float(os.getenv("DEDUP_COSINE_THRESHOLD", "0.78")),
        decision_threshold=float(os.getenv("DEDUP_DECISION_THRESHOLD", "0.80")),
        embedding_model=os.getenv("DEDUP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        decision_model=os.getenv("DEDUP_DECISION_MODEL", "cross-encoder/stsb-distilroberta-base"),
    )


def _write_ingested(document, extract: bool, model_name: str, store, model=None, semantic=None) -> None:
    store.write_document(document)
    if extract:
        store.write_extraction(
            document,
            _extract(document, model_name, store.read_concepts(), semantic or _semantic_deduplicator(), model or _load_ner_model(model_name)),
        )


def _ingest(document, extract: bool, model_name: str) -> None:
    with _store() as store:
        store.ensure_schema()
        _write_ingested(document, extract, model_name, store)


def _crawl_openalex(query: str, limit: int, per_page: int, checkpoint: Path, extract: bool, model_name: str) -> None:
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    if state and state.get("query") != query:
        raise RuntimeError(f"Checkpoint belongs to another query: {state['query']!r}")
    processed = int(state.get("processed", 0))
    cursor = state.get("cursor", "*")
    started = perf_counter()
    model = _load_ner_model(model_name) if extract else None
    semantic = _semantic_deduplicator() if extract else None
    with _store() as store:
        store.ensure_schema()
        while processed < limit and cursor:
            page = fetch_openalex_page(query, cursor, min(per_page, limit - processed), os.getenv("OPENALEX_MAILTO"))
            works = page.get("results", [])
            if not works:
                break
            for payload in works:
                document = parse_openalex(payload, raw=json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
                _write_ingested(document, extract, model_name, store, model, semantic)
                processed += 1
            cursor = page.get("meta", {}).get("next_cursor")
            checkpoint.write_text(json.dumps({"query": query, "processed": processed, "cursor": cursor}), encoding="utf-8")
            elapsed = perf_counter() - started
            print(f"processed={processed}/{limit} elapsed={elapsed:.1f}s rate={processed / elapsed:.2f} works/s", flush=True)


def _uniform_sample(names: List[str], limit: int) -> List[str]:
    if limit > len(names):
        raise RuntimeError(f"PyPI has only {len(names)} projects")
    return [names[index * len(names) // limit] for index in range(limit)]


def _crawl_pypi(limit: int, checkpoint: Path, extract: bool, model_name: str) -> None:
    state = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    packages = state.get("packages") or _uniform_sample(fetch_pypi_projects(), limit)
    processed = int(state.get("processed", 0))
    failures = int(state.get("failures", 0))
    started = perf_counter()
    model = _load_ner_model(model_name) if extract else None
    semantic = _semantic_deduplicator() if extract else None
    with _store() as store:
        store.ensure_schema()
        for package in packages[processed:]:
            try:
                payload = fetch_pypi(package)
                document = parse_pypi(payload, raw=json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
                _write_ingested(document, extract, model_name, store, model, semantic)
            except Exception as exc:
                failures += 1
                print(f"skipped={package} error={exc}", flush=True)
            processed += 1
            if processed % 25 == 0 or processed == len(packages):
                checkpoint.write_text(
                    json.dumps({"packages": packages, "processed": processed, "failures": failures}),
                    encoding="utf-8",
                )
                elapsed = perf_counter() - started
                print(f"processed={processed}/{len(packages)} failures={failures} elapsed={elapsed:.1f}s rate={processed / elapsed:.2f} packages/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="lctrend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_command = subparsers.add_parser("parse", help="Parse a saved source response")
    parse_command.add_argument("kind", choices=[*PARSERS, "epo"])
    parse_command.add_argument("input", type=Path)
    parse_command.add_argument("--output", type=Path)

    fetch_command = subparsers.add_parser("fetch", help="Fetch and parse a public API record")
    fetch_command.add_argument("kind", choices=["openalex", "github", "pypi"])
    fetch_command.add_argument("identifier")
    fetch_command.add_argument("--output", type=Path)
    fetch_command.add_argument("--ingest", action="store_true")
    fetch_command.set_defaults(extract=True)
    fetch_command.add_argument("--extract", dest="extract", action="store_true", help=argparse.SUPPRESS)
    fetch_command.add_argument("--no-extract", dest="extract", action="store_false", help="Skip GLiNER during ingest")
    fetch_command.add_argument(
        "--ner-model", default=os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
    )

    ingest_command = subparsers.add_parser("ingest", help="Parse a saved record into Neo4j")
    ingest_command.add_argument("kind", choices=[*PARSERS, "epo"])
    ingest_command.add_argument("input", type=Path)
    ingest_command.set_defaults(extract=True)
    ingest_command.add_argument("--extract", dest="extract", action="store_true", help=argparse.SUPPRESS)
    ingest_command.add_argument("--no-extract", dest="extract", action="store_false", help="Skip GLiNER during ingest")
    ingest_command.add_argument(
        "--ner-model", default=os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
    )

    crawl_command = subparsers.add_parser("crawl-openalex", help="Crawl OpenAlex search results into Neo4j")
    crawl_command.add_argument("query", nargs="?", default="artificial intelligence")
    crawl_command.add_argument("--limit", type=int, default=500)
    crawl_command.add_argument("--per-page", type=int, default=100)
    crawl_command.add_argument("--checkpoint", type=Path, default=Path(".openalex-crawl.json"))
    crawl_command.set_defaults(extract=True)
    crawl_command.add_argument("--no-extract", dest="extract", action="store_false")
    crawl_command.add_argument(
        "--ner-model", default=os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
    )

    pypi_crawl_command = subparsers.add_parser("crawl-pypi", help="Crawl a uniform PyPI sample into Neo4j")
    pypi_crawl_command.add_argument("--limit", type=int, default=5000)
    pypi_crawl_command.add_argument("--checkpoint", type=Path, default=Path(".pypi-crawl.json"))
    pypi_crawl_command.set_defaults(extract=True)
    pypi_crawl_command.add_argument("--no-extract", dest="extract", action="store_false")
    pypi_crawl_command.add_argument(
        "--ner-model", default=os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
    )

    subparsers.add_parser("init-graph", help="Create Neo4j constraints")
    args = parser.parse_args()

    if args.command == "init-graph":
        with _store() as store:
            store.ensure_schema()
        return

    if args.command == "crawl-openalex":
        _crawl_openalex(args.query, args.limit, args.per_page, args.checkpoint, args.extract, args.ner_model)
        return

    if args.command == "crawl-pypi":
        _crawl_pypi(args.limit, args.checkpoint, args.extract, args.ner_model)
        return

    if args.command == "fetch":
        fetchers = {
            "openalex": lambda: fetch_openalex(args.identifier, os.getenv("OPENALEX_MAILTO")),
            "github": lambda: fetch_github(args.identifier, os.getenv("GITHUB_TOKEN")),
            "pypi": lambda: fetch_pypi(args.identifier),
        }
        payload: Dict[str, Any] = fetchers[args.kind]()
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        document = PARSERS[args.kind](payload, raw=raw)
        if args.output:
            args.output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        else:
            print(document.model_dump_json(indent=2))
        if args.ingest:
            _ingest(document, args.extract, args.ner_model)
        return

    document = _parse(args.kind, args.input)
    if args.command == "parse":
        if args.output:
            args.output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        else:
            print(document.model_dump_json(indent=2))
    else:
        _ingest(document, args.extract, args.ner_model)
