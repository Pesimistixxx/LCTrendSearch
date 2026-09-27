"""Offline CLI integration checks, with an in-memory graph boundary."""

import hashlib
import json
import os
import sys

import pytest

from lctrend import cli
from lctrend.ingest.adapters import parse_openalex
from lctrend.llm.client import JsonLLM, LLMError, ReplayProvider


class MemoryStore:
    def __init__(self):
        self.events = []
        self.documents = []
        self.extractions = []

    def __enter__(self):
        self.events.append("open")
        return self

    def __exit__(self, *args):
        self.events.append("close")

    def ensure_schema(self):
        self.events.append("schema")

    def read_concepts(self):
        self.events.append("registry")
        return []

    def write_document(self, document):
        self.events.append("document")
        self.documents.append(document)

    def write_extraction(self, document, result):
        self.events.append("extraction")
        self.extractions.append(result)

    def write_processed(self, document, result):
        self.events.append("processed")
        self.documents.append(document)
        self.extractions.append(result)


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    for name in (
        "LLM_MODEL",
        "LLM_EXTRACT_MODEL",
        "LLM_REVIEW_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LCTREND_CONFIG_DIR",
        "LLM_PROVIDER",
        "LLM_MODEL_LADDER",
        "LLM_CA_BUNDLE_FILE",
        "LCTREND_EXTRACTOR",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_SCOPE",
        "GIGACHAT_BASE_URL",
        "GIGACHAT_AUTH_URL",
        "GIGACHAT_CA_BUNDLE_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    store = MemoryStore()
    monkeypatch.setattr(cli, "_store", lambda: store)

    return store


def record(tmp_path):
    payload = {
        "id": "https://openalex.org/W101",
        "title": "Local fixture",
        "abstract_inverted_index": {
            "Original": [0],
            "source": [1],
            "text": [2],
        },
    }
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--extractor", "llm"],
    ],
)
def test_ingest_defaults_to_llm(
    monkeypatch, tmp_path, offline, arguments
):
    source = record(tmp_path)
    output = tmp_path / "nested" / "extraction.json"
    provider = ReplayProvider(
        [{"stage": "extract", "response": {"entities": [], "claims": []}}]
    )
    constructors = []

    def configured(**kwargs):
        constructors.append(kwargs)
        return provider

    monkeypatch.setattr(JsonLLM, "from_environment", configured)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lctrend",
            "ingest",
            "openalex",
            str(source),
            "--extraction-output",
            str(output),
            *arguments,
        ],
    )
    cli.main()
    assert len(constructors) == 1
    assert len(provider.calls) == 1 and provider.calls[0]["stage"] == "extract"
    assert offline.events == [
        "open",
        "schema",
        "registry",
        "processed",
        "close",
    ]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert "ner" not in saved["run"]["metadata"]
    assert (
        saved["document_version_id"]
        == offline.documents[0].document_version_id
    )
    assert saved["run"]["parser"] == "llm_packets"
    assert saved["run"]["metadata"]["demo"] is True
    assert (
        saved["run"]["metadata"]["provider_calls"][0]["provider"] == "replay"
    )
    assert saved["run"]["metadata"]["coverage"]["unprocessed_chunk_ids"] == []


def test_no_extract_ingest_never_constructs_provider(
    monkeypatch, tmp_path, offline
):
    def prohibited(**kwargs):
        pytest.fail("--no-extract must not initialize an LLM")

    monkeypatch.setattr(JsonLLM, "from_environment", prohibited)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lctrend",
            "ingest",
            "openalex",
            str(record(tmp_path)),
            "--no-extract",
            "--extractor",
            "llm",
        ],
    )
    cli.main()
    assert offline.events == ["open", "schema", "document", "close"]
    assert offline.extractions == []


def test_parse_only_never_initializes_llm_or_graph(
    monkeypatch, tmp_path, offline
):
    def prohibited(*args, **kwargs):
        pytest.fail("parse does not initialize extraction or graph")

    monkeypatch.setattr(JsonLLM, "from_environment", prohibited)
    monkeypatch.setattr(cli, "_store", prohibited)
    output = tmp_path / "document.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lctrend",
            "parse",
            "openalex",
            str(record(tmp_path)),
            "--output",
            str(output),
        ],
    )
    cli.main()
    assert (
        json.loads(output.read_text(encoding="utf-8"))["chunks"][0]["text"]
        == "Original source text"
    )


@pytest.mark.parametrize(
    "ingest,extract", [(False, False), (True, True), (True, False)]
)
def test_fetch_only_and_fetch_ingest_respect_extraction_switch(
    monkeypatch, tmp_path, offline, ingest, extract
):
    monkeypatch.setattr(
        cli,
        "fetch_openalex",
        lambda *args: {
            "id": "https://openalex.org/W1",
            "title": "Fixture",
            "abstract_inverted_index": {"Source": [0], "text": [1]},
        },
    )
    provider = ReplayProvider(
        [{"stage": "extract", "response": {"entities": [], "claims": []}}]
    )
    constructors = []

    def configured(**kwargs):
        constructors.append(kwargs)
        return provider

    monkeypatch.setattr(JsonLLM, "from_environment", configured)
    arguments = [
        "lctrend",
        "fetch",
        "openalex",
        "W1",
        "--output",
        str(tmp_path / "fetched.json"),
    ]
    if ingest:
        arguments.append("--ingest")
        if not extract:
            arguments.append("--no-extract")
    monkeypatch.setattr(sys, "argv", arguments)
    cli.main()
    assert len(constructors) == int(extract)
    assert len(offline.documents) == int(ingest)
    assert len(offline.extractions) == int(extract)
    assert len(provider.calls) == int(extract)


def test_provider_configuration_fails_before_any_graph_writes(
    tmp_path, offline
):
    document = parse_openalex(
        {"id": "https://openalex.org/W1", "title": "Fixture"}
    )
    with pytest.raises(LLMError) as failure:
        cli._ingest(document, True)
    assert failure.value.code == "configuration"
    assert offline.events == []
    assert offline.documents == [] and offline.extractions == []


@pytest.mark.parametrize("crawler", ["openalex", "pypi"])
def test_crawl_provider_configuration_fails_before_graph_writes_or_fetch(
    monkeypatch, tmp_path, offline, crawler
):
    def no_fetch(*args, **kwargs):
        pytest.fail(
            "Configuration failure should precede source fetch "
            "with an explicit package list"
        )

    monkeypatch.setattr(cli, "fetch_openalex_page", no_fetch)
    monkeypatch.setattr(cli, "fetch_pypi", no_fetch)
    with pytest.raises(LLMError) as failure:
        if crawler == "openalex":
            cli._crawl_openalex(
                "fixture", 1, 1, tmp_path / "oa.json", True
            )
        else:
            cli._crawl_pypi(
                1,
                tmp_path / "pypi.json",
                True,
                requested_packages=["fixture"],
            )
    assert failure.value.code == "configuration"
    assert offline.events == []


@pytest.mark.parametrize("command", ["crawl-openalex", "crawl-pypi"])
@pytest.mark.parametrize(
    "extra,extract,extractor",
    [
        ([], True, "llm"),
        (["--extractor", "llm"], True, "llm"),
        (["--no-extract"], False, "llm"),
    ],
)
def test_crawler_cli_passes_extraction_choice(
    monkeypatch, command, extra, extract, extractor
):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.delenv("LCTREND_EXTRACTOR", raising=False)
    received = []
    monkeypatch.setattr(
        cli, "_crawl_openalex", lambda *args, **kwargs: received.append(args)
    )
    monkeypatch.setattr(
        cli, "_crawl_pypi", lambda *args: received.append(args)
    )
    arguments = ["lctrend", command]
    if command == "crawl-openalex":
        arguments.append("fixture-query")
    monkeypatch.setattr(sys, "argv", [*arguments, *extra])
    cli.main()
    args = received[0]
    assert args[4 if command == "crawl-openalex" else 2] is extract
    assert args[-1] == extractor


def test_default_extractor_is_llm(monkeypatch):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.delenv("LCTREND_CONFIG_DIR", raising=False)
    received = []
    monkeypatch.setattr(
        cli, "_crawl_openalex", lambda *args, **kwargs: received.append(args)
    )
    monkeypatch.setattr(sys, "argv", ["lctrend", "crawl-openalex", "robotics"])
    cli.main()
    assert received[0][5] == "llm"


def test_snapshot_is_content_addressed_and_repeated_bytes_are_not_rewritten(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    raw = b'{ "id": "W1", "original": "exact whitespace\r\n" }'
    document = parse_openalex(
        {"id": "https://openalex.org/W1", "title": "Fixture"}
    )
    identity = document.document_version_id
    media_type = document.artifact.media_type
    cli._snapshot(document, raw)
    path = tmp_path / "raw" / hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert document.artifact.sha256 == path.name
    assert document.artifact.byte_length == len(raw)
    assert document.artifact.media_type == media_type
    assert document.artifact.uri == path.resolve().as_uri()
    assert document.document_version_id == identity
    os.utime(path, (1_600_000_000, 1_600_000_000))
    original_mtime = path.stat().st_mtime_ns
    cli._snapshot(document, raw)
    assert path.stat().st_mtime_ns == original_mtime
    assert len(list((tmp_path / "raw").iterdir())) == 1


def test_snapshot_refuses_to_overwrite_content_hash_collision(
    tmp_path, monkeypatch
):
    directory = tmp_path / "raw"
    directory.mkdir()
    monkeypatch.setenv("LCTREND_RAW_DIR", str(directory))
    raw = b"original source bytes"
    path = directory / hashlib.sha256(raw).hexdigest()
    path.write_bytes(b"damaged existing snapshot")
    document = parse_openalex(
        {"id": "https://openalex.org/W1", "title": "Fixture"}
    )
    before = document.artifact.model_dump()
    with pytest.raises(ValueError, match="content hash"):
        cli._snapshot(document, raw)
    assert path.read_bytes() == b"damaged existing snapshot"
    assert document.artifact.model_dump() == before
