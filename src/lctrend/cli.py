from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .adapters import parse_epo, parse_github, parse_openalex, parse_pypi
from .connectors import fetch_github, fetch_openalex, fetch_pypi
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


def _extract(document, model_name: str, registry, semantic: SemanticDeduplicator) -> ExtractionResult:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError('Install NER support first: pip install -e ".[ner]"') from exc

    model = GLiNER.from_pretrained(model_name)
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


def _ingest(document, extract: bool, model_name: str) -> None:
    with _store() as store:
        store.ensure_schema()
        store.write_document(document)
        if extract:
            semantic = SemanticDeduplicator(
                cosine_threshold=float(os.getenv("DEDUP_COSINE_THRESHOLD", "0.78")),
                decision_threshold=float(os.getenv("DEDUP_DECISION_THRESHOLD", "0.80")),
                embedding_model=os.getenv(
                    "DEDUP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                decision_model=os.getenv(
                    "DEDUP_DECISION_MODEL", "cross-encoder/stsb-distilroberta-base"
                ),
            )
            store.write_extraction(
                document, _extract(document, model_name, store.read_concepts(), semantic)
            )


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
    fetch_command.add_argument("--extract", action="store_true", help="Run GLiNER during ingest")
    fetch_command.add_argument(
        "--ner-model", default=os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
    )

    ingest_command = subparsers.add_parser("ingest", help="Parse a saved record into Neo4j")
    ingest_command.add_argument("kind", choices=[*PARSERS, "epo"])
    ingest_command.add_argument("input", type=Path)
    ingest_command.add_argument("--extract", action="store_true", help="Run GLiNER during ingest")
    ingest_command.add_argument(
        "--ner-model", default=os.getenv("GLINER_MODEL", "urchade/gliner_medium-v2.1")
    )

    subparsers.add_parser("init-graph", help="Create Neo4j constraints")
    args = parser.parse_args()

    if args.command == "init-graph":
        with _store() as store:
            store.ensure_schema()
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
