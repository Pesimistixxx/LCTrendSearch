import asyncio
import gzip
import json
import logging
import zlib
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from lctrend.ingest import connectors


@pytest.mark.parametrize(
    ("encoding", "compress"),
    [("gzip", gzip.compress), ("deflate", zlib.compress)],
)
def test_request_decodes_compressed_stream_once(
    monkeypatch, encoding, compress
):
    payload = {"results": [{"title": "Машинное обучение"}]}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    compressed = compress(body)

    def fetch(request):
        return httpx.Response(
            200,
            headers={
                "Content-Encoding": encoding,
                "Content-Length": str(len(compressed)),
                "Content-Type": "application/json",
                "X-RateLimit-Remaining": "10",
            },
            stream=httpx.ByteStream(compressed),
        )

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(fetch))
    response = asyncio.run(connectors.request("https://api.openalex.org/works"))
    assert response.content == body
    assert response.json() == payload
    assert "Content-Encoding" not in response.headers
    assert int(response.headers["Content-Length"]) == len(body)
    assert response.headers["Content-Type"] == "application/json"
    assert response.headers["X-RateLimit-Remaining"] == "10"


def test_compressed_response_limit_applies_to_decoded_body(monkeypatch):
    body = b"x" * 1000
    compressed = gzip.compress(body)
    assert len(compressed) < 100

    monkeypatch.setattr(
        connectors,
        "TRANSPORT",
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=httpx.ByteStream(compressed),
            )
        ),
    )
    with pytest.raises(ValueError, match="size limit"):
        asyncio.run(
            connectors.request("https://example.org/data", max_bytes=100)
        )


def test_openalex_page_includes_search_and_cursor(monkeypatch):
    captured = {}

    def fake_fetch(url, headers=None):
        captured["url"] = url
        return {"results": []}

    monkeypatch.setattr(connectors, "fetch_json", fake_fetch)
    asyncio.run(
        connectors.fetch_openalex_page(
            "edge computing", "next token", 25, "me@example.com"
        )
    )
    assert "search=edge+computing" in captured["url"]
    assert "cursor=next+token" in captured["url"]
    assert "per-page=25" in captured["url"]


def test_pypi_projects_reads_json_simple_index(monkeypatch):
    monkeypatch.setattr(
        connectors,
        "fetch_json",
        lambda url, headers=None: {
            "projects": [{"name": "one"}, {"name": "two"}]
        },
    )
    assert asyncio.run(connectors.fetch_pypi_projects()) == ["one", "two"]


def test_github_fetches_commit_then_reads_same_sha(monkeypatch):
    calls = []

    def fake_fetch(url, headers=None):
        calls.append(url)
        if url.endswith("/repos/org/repo"):
            return {"full_name": "org/repo", "default_branch": "feature/main"}
        if "/commits/" in url:
            return {
                "sha": "fixedsha",
                "commit": {"committer": {"date": "2025-01-01"}},
            }
        if "/readme?" in url:
            return {"text": "README"}
        return []

    monkeypatch.setattr(connectors, "fetch_json", fake_fetch)
    payload = asyncio.run(connectors.fetch_github("org/repo"))
    assert calls[1].endswith("/commits/feature%2Fmain")
    assert calls[2].endswith("/readme?ref=fixedsha")
    assert payload["commit"]["sha"] == "fixedsha"
    assert payload["_retrieved_at"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" w123 ", "W123"),
        ("works/W123", "W123"),
        ("https://openalex.org/W123", "W123"),
        ("https://openalex.org/works/w123?view=full#authors", "W123"),
        ("https://api.openalex.org/works/W123", "W123"),
        ("10.1234/ABC", "doi:10.1234/abc"),
        ("doi:10.1234/ABC", "doi:10.1234/abc"),
        ("https://doi.org/10.1234/ABC", "doi:10.1234/abc"),
        ("http://dx.doi.org/10.1234/ABC", "doi:10.1234/abc"),
        ("PMID:123", "pmid:123"),
    ],
)
def test_normalizes_openalex_work_identifiers(value, expected):
    assert connectors.normalize_openalex_work_id(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "A123",
        "https://example.org/W123",
        "https://openalex.org.evil.example/W123",
        "https://user:password@openalex.org/W123",
        "W123?api_key=secret",
    ],
)
def test_invalid_openalex_identifiers_are_rejected(value):
    with pytest.raises(ValueError):
        connectors.normalize_openalex_work_id(value)


def test_single_work_escapes_doi_reserved_characters(monkeypatch):
    captured = {}

    def fetch(url):
        captured["url"] = url
        return {"id": "https://openalex.org/W1"}

    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setattr(connectors, "fetch_json", fetch)
    payload = asyncio.run(
        connectors.fetch_openalex("10.1234/a&b?c#d", "me@example.com")
    )
    parsed = urlsplit(captured["url"])
    assert parsed.path.endswith("/doi:10.1234/a%26b%3Fc%23d")
    assert parse_qs(parsed.query) == {"mailto": ["me@example.com"]}
    assert payload["_retrieved_at"]


@pytest.mark.parametrize("page", [False, 0, 101, 200, 1.5, "25"])
def test_openalex_page_rejects_unsupported_size(page):
    with pytest.raises(ValueError, match="per_page"):
        asyncio.run(connectors.fetch_openalex_page("sensors", per_page=page))


@pytest.mark.parametrize("page", [False, True])
def test_openalex_env_key_uses_header_and_never_logs_secret(
    monkeypatch, caplog, page
):
    calls = []

    def fetch(request):
        calls.append(request)
        payload = (
            {"results": [{"id": "https://openalex.org/W1"}]}
            if page
            else {"id": "https://openalex.org/W1"}
        )
        return httpx.Response(200, json=payload)

    monkeypatch.setenv("OPENALEX_API_KEY", "test-private-key")
    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(fetch))
    caplog.set_level(logging.DEBUG)
    if page:
        result = asyncio.run(connectors.fetch_openalex_page("sensors"))
        assert result["results"][0]["_retrieved_at"]
    else:
        asyncio.run(connectors.fetch_openalex("W1"))
    assert calls[0].headers["Authorization"] == "Bearer test-private-key"
    assert "test-private-key" not in str(calls[0].url)
    assert "test-private-key" not in caplog.text


def test_explicit_openalex_key_overrides_env_and_can_disable_it(monkeypatch):
    captured = []

    def fetch(url, headers=None):
        captured.append(headers)
        return {"id": "https://openalex.org/W1"}

    monkeypatch.setenv("OPENALEX_API_KEY", "env-key")
    monkeypatch.setattr(connectors, "fetch_json", fetch)
    asyncio.run(connectors.fetch_openalex("W1", api_key="explicit-key"))
    asyncio.run(connectors.fetch_openalex("W1", api_key=""))
    assert captured == [{"Authorization": "Bearer explicit-key"}, None]


@pytest.mark.parametrize(
    "payload", [{}, {"results": None}, {"results": [None]}]
)
def test_openalex_page_rejects_invalid_work_list(monkeypatch, payload):
    monkeypatch.setattr(connectors, "fetch_json", lambda *args: payload)
    with pytest.raises(ValueError, match="list of works"):
        asyncio.run(connectors.fetch_openalex_page("sensors"))


def test_openalex_rate_limit_reset_is_a_duration():
    response = httpx.Response(
        429,
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "30"},
        request=httpx.Request("GET", "https://api.openalex.org/works"),
    )
    assert connectors._retry_after(response) == 31
