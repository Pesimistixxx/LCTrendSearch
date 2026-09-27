"""OpenAlex CLI through the real connector/parser and a graph boundary."""

import json
import sys

import httpx
import pytest

from lctrend import cli
from lctrend.ingest import connectors


class DocumentStore:
    def __init__(self):
        self.documents = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def ensure_schema(self):
        pass

    def write_document(self, document):
        self.documents.append(document)

    def write_crawl_run(self, run):
        pass


@pytest.fixture
def fast_ingestion(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_environment", lambda: None)
    monkeypatch.setenv("OPENALEX_API_KEY", "offline-openalex-key")
    monkeypatch.setenv("OPENALEX_MAILTO", "research@example.org")
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.delenv("LCTREND_CONFIG_DIR", raising=False)
    store = DocumentStore()
    monkeypatch.setattr(cli, "_store", lambda: store)

    def unexpected(*args, **kwargs):
        pytest.fail("Fast metadata ingestion must not load PDF or models")

    monkeypatch.setattr(cli, "require_pdf_support", unexpected)
    monkeypatch.setattr(cli, "_load_ner_model", unexpected)
    monkeypatch.setattr(cli, "_semantic_deduplicator", unexpected)
    return store


def work(number):
    return {
        "id": f"https://openalex.org/W{number}",
        "doi": f"https://doi.org/10.1234/work{number}",
        "title": f"Paper {number}",
        "abstract_inverted_index": {"Sensor": [0], "research": [1]},
    }


def test_fast_crawl_authenticates_and_resumes_without_reimport(
    monkeypatch, tmp_path, fast_ingestion
):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.headers["Authorization"] == (
            "Bearer offline-openalex-key"
        )
        assert "offline-openalex-key" not in str(request.url)
        assert request.url.params["search"] == "edge computing"
        assert request.url.params["mailto"] == "research@example.org"
        assert request.url.params["filter"] == "has_abstract:true"
        cursor = request.url.params["cursor"]
        number = 1 if cursor == "*" else 2
        return httpx.Response(
            200,
            json={
                "results": [work(number)],
                "meta": {"next_cursor": "second" if number == 1 else None},
            },
        )

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(respond))
    checkpoint = tmp_path / "crawl.json"

    def run(limit):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lctrend", "crawl-openalex", "edge computing",
                "--limit", str(limit), "--per-page", "1",
                "--checkpoint", str(checkpoint),
                "--filter", "has_abstract:true",
                "--no-fulltext", "--no-extract",
            ],
        )
        cli.main()

    run(1)
    assert json.loads(checkpoint.read_text())["cursor"] == "second"
    run(2)
    run(2)
    assert [request.url.params["cursor"] for request in requests] == [
        "*", "second",
    ]
    assert [doc.source.record_id for doc in fast_ingestion.documents] == [
        "W1", "W2",
    ]
    assert all(
        doc.coverage == "abstract_only"
        and doc.chunks[0].text == "Sensor research"
        and doc.source.source_id == "source:openalex"
        and "offline-openalex-key" not in doc.model_dump_json()
        for doc in fast_ingestion.documents
    )
    assert json.loads(checkpoint.read_text())["processed"] == 2


@pytest.mark.parametrize("per_page", [0, 101, 200])
def test_crawl_rejects_unsupported_page_sizes(tmp_path, per_page):
    with pytest.raises(ValueError, match="per-page must be 1..100"):
        cli._crawl_openalex(
            "sensors", 1, per_page, tmp_path / "crawl.json", False, "unused",
            fulltext=False,
        )


def test_empty_page_marks_checkpoint_complete(
    monkeypatch, tmp_path, fast_ingestion
):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json={"results": [], "meta": {"next_cursor": "stale"}}
        )

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(respond))
    checkpoint = tmp_path / "empty.json"
    for _ in range(2):
        cli._crawl_openalex(
            "sensors", 3, 1, checkpoint, False, "unused", fulltext=False
        )
    assert len(requests) == 1
    assert json.loads(checkpoint.read_text())["cursor"] is None
    assert fast_ingestion.documents == []


@pytest.mark.parametrize("limit", [1, 3])
def test_repeated_cursor_stops_crawl_without_duplicate_import(
    monkeypatch, tmp_path, fast_ingestion, limit
):
    monkeypatch.setattr(
        connectors, "TRANSPORT",
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"results": [work(1)], "meta": {"next_cursor": "*"}}
            )
        ),
    )
    with pytest.raises(ValueError, match="Repeated OpenAlex cursor"):
        cli._crawl_openalex(
            "sensors", limit, 1, tmp_path / "loop.json", False, "unused",
            fulltext=False,
        )
    assert fast_ingestion.documents == []
    assert not (tmp_path / "loop.json").exists()


def test_missing_cursor_preserves_checkpoint_for_retry(
    monkeypatch, tmp_path, fast_ingestion
):
    checkpoint = tmp_path / "retry.json"
    saved = {
        "query": "sensors", "filter": None, "processed": 1,
        "failures": 0, "cursor": "second",
    }
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(
        connectors, "TRANSPORT",
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"results": [work(2)], "meta": {}}
            )
        ),
    )
    with pytest.raises(ValueError, match="missing next_cursor"):
        cli._crawl_openalex(
            "sensors", 2, 1, checkpoint, False, "unused", fulltext=False
        )
    assert json.loads(checkpoint.read_text()) == saved
    assert fast_ingestion.documents == []
