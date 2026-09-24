from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from .adapters import parse_epo, parse_github, parse_openalex, parse_pypi
from .connectors import fetch_github, fetch_openalex, fetch_pypi
from .graph import GraphStore


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

    ingest_command = subparsers.add_parser("ingest", help="Parse a saved record into Neo4j")
    ingest_command.add_argument("kind", choices=[*PARSERS, "epo"])
    ingest_command.add_argument("input", type=Path)

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
            with _store() as store:
                store.ensure_schema()
                store.write_document(document)
        return

    document = _parse(args.kind, args.input)
    if args.command == "parse":
        if args.output:
            args.output.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        else:
            print(document.model_dump_json(indent=2))
    else:
        with _store() as store:
            store.ensure_schema()
            store.write_document(document)
