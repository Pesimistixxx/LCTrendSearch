import asyncio
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from lctrend.ingest import connectors, fulltext
from lctrend.ingest.adapters import parse_openalex
from lctrend.ingest.file_adapters import UnsupportedFileFormat

WORK = {
    "id": "https://openalex.org/W1",
    "title": "Fixture",
    "abstract_inverted_index": {"Short": [0], "abstract.": [1]},
    "best_oa_location": {
        "is_oa": True,
        "pdf_url": "https://example.org/paper.pdf",
    },
    "primary_location": {
        "is_oa": False,
        "pdf_url": "https://publisher.example/paywalled.pdf",
    },
    "locations": [
        {"is_oa": True, "pdf_url": "https://example.org/paper.pdf"},
        {"is_oa": True, "pdf_url": "https://arxiv.example/mirror.pdf"},
        {"is_oa": True, "pdf_url": "file:///etc/passwd"},
    ],
}


@pytest.fixture
def docling(monkeypatch, tmp_path):
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    items = [
        ("page_header", "Journal of Fixtures, vol. 1"),
        ("section_header", "1 Introduction"),
        ("text", "Sparse attention reduces memory use by 40%."),
        ("section_header", "References"),
        ("list_item", "[1] Someone. A cited paper. 2020."),
        ("section_header", "Appendix A"),
        ("text", "Additional experiments."),
    ]
    document = SimpleNamespace(
        iterate_items=lambda: [
            (
                SimpleNamespace(
                    label=SimpleNamespace(value=label),
                    text=text,
                    self_ref=f"#/texts/{index}",
                    prov=[],
                ),
                0,
            )
            for index, (label, text) in enumerate(items)
        ]
    )

    class Converter:
        def convert(self, path):
            return SimpleNamespace(
                document=document, status=SimpleNamespace(value="success")
            )

    module = ModuleType("docling.document_converter")
    module.DocumentConverter = Converter
    monkeypatch.setitem(sys.modules, "docling.document_converter", module)
    monkeypatch.setattr(fulltext, "require_pdf_support", lambda: None)


def test_pdf_candidates_are_open_access_http_links_in_openalex_order():
    assert fulltext.openalex_pdf_urls(WORK) == [
        "https://example.org/paper.pdf",
        "https://arxiv.example/mirror.pdf",
    ]


def test_fulltext_is_appended_after_abstract_without_headers_or_bibliography(
    docling,
):
    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(WORK), WORK, fetch=lambda url: b"%PDF-1.7 fixture"
        )
    )
    texts = [chunk.text for chunk in document.chunks]
    assert texts[0] == "Short abstract."
    assert texts[1:] == [
        "1 Introduction",
        "Sparse attention reduces memory use by 40%.",
        "Appendix A",
        "Additional experiments.",
    ]
    body = document.chunks[1:]
    assert {chunk.kind for chunk in body} == {"fulltext"}
    assert [chunk.order for chunk in document.chunks] == list(
        range(len(document.chunks))
    )
    assert body[1].locator["section_heading"] == "1 Introduction"
    assert body[1].locator["docling_label"] == "text"
    assert document.coverage == "abstract_and_full_text"
    assert document.metadata["fulltext"]["status"] == "parsed"
    assert (
        document.metadata["fulltext"]["pdf_url"]
        == "https://example.org/paper.pdf"
    )


def test_next_candidate_is_tried_and_failures_are_recorded(docling):
    def fetch(url):
        if "example.org" in url:
            raise ValueError("URL did not return a PDF")
        return b"%PDF-1.7 mirror"

    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(WORK), WORK, fetch=fetch
        )
    )
    status = document.metadata["fulltext"]
    assert status["pdf_url"] == "https://arxiv.example/mirror.pdf"
    assert status["attempts"][0]["url"] == "https://example.org/paper.pdf"


def test_work_without_pdf_stays_abstract_only_without_docling(monkeypatch):
    monkeypatch.setattr(
        fulltext,
        "require_pdf_support",
        lambda: pytest.fail("no PDF, no Docling needed"),
    )
    work = {
        key: value
        for key, value in WORK.items()
        if key not in ("best_oa_location", "primary_location", "locations")
    }
    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(work),
            work,
            fetch=lambda url: pytest.fail("nothing to fetch"),
        )
    )
    assert document.coverage == "abstract_only"
    assert document.metadata["fulltext"]["status"] == "no_pdf_url"


def test_missing_docling_is_explicit(monkeypatch):
    monkeypatch.setattr(
        fulltext.importlib.util, "find_spec", lambda name: None
    )
    with pytest.raises(UnsupportedFileFormat, match="--no-fulltext"):
        asyncio.run(
            fulltext.attach_openalex_fulltext(
                parse_openalex(WORK), WORK, fetch=lambda url: b"%PDF-"
            )
        )


def test_fetch_pdf_rejects_landing_pages_and_non_http(monkeypatch):
    monkeypatch.setattr(
        connectors,
        "TRANSPORT",
        httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=b"<html>captcha</html>"
            )
        ),
    )
    with pytest.raises(ValueError, match="did not return a PDF"):
        asyncio.run(connectors.fetch_pdf("https://example.org/paper.pdf"))
    with pytest.raises(ValueError, match="HTTP"):
        asyncio.run(connectors.fetch_pdf("file:///etc/passwd"))


def test_transient_source_errors_are_retried_with_backoff(monkeypatch):
    answers = [
        httpx.Response(503),
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"results": []}),
    ]
    requests, pauses = [], []

    def respond(request):
        requests.append(request)
        return answers.pop(0)

    async def no_sleep(delay):
        pauses.append(delay)

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(respond))
    monkeypatch.setattr(connectors.asyncio, "sleep", no_sleep)
    assert asyncio.run(connectors.fetch_json("https://api.example/works")) == {
        "results": []
    }
    assert len(requests) == 3
    assert pauses[1] == 0  # Retry-After is honoured


def test_permanent_source_error_is_not_retried(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(404)

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(respond))
    with pytest.raises(connectors.SourceHTTPError, match="404"):
        asyncio.run(connectors.fetch_json("https://api.example/missing"))
    assert len(requests) == 1


def test_github_rate_limit_waits_for_reset(monkeypatch):
    answers = [
        httpx.Response(
            403,
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(connectors.time.time()) + 30),
            },
        ),
        httpx.Response(200, json={"full_name": "org/repo"}),
    ]
    pauses = []

    async def no_sleep(delay):
        pauses.append(delay)

    monkeypatch.setattr(
        connectors,
        "TRANSPORT",
        httpx.MockTransport(lambda request: answers.pop(0)),
    )
    monkeypatch.setattr(connectors.asyncio, "sleep", no_sleep)
    assert asyncio.run(
        connectors.fetch_json("https://api.github.com/repos/org/repo")
    ) == {"full_name": "org/repo"}
    assert 25 <= pauses[0] <= 32


def test_openalex_page_passes_filter(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        connectors,
        "fetch_json",
        lambda url, headers=None: (
            captured.setdefault("url", url) and {"results": []}
        ),
    )
    asyncio.run(
        connectors.fetch_openalex_page(
            "edge computing", "*", 10, None, "is_oa:true,has_abstract:true"
        )
    )
    assert "filter=is_oa%3Atrue%2Chas_abstract%3Atrue" in captured["url"]


def test_section_headings_get_a_rhetorical_role():
    assert fulltext.section_role("3. Experimental setup") == "method"
    assert fulltext.section_role("Limitations") == "limitations"
    assert fulltext.section_role("Заключение") == "conclusion"
    assert fulltext.section_role(None) is None
