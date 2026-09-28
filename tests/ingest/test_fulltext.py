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

PUBMED_WORK = {
    "id": "https://openalex.org/W2908201961",
    "doi": "https://doi.org/10.1038/s41591-018-0300-7",
    "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/30617339"},
}
PUBMED_XML = (
    b"<PubmedArticleSet><PubmedArticle><MedlineCitation>"
    b"<PMID>30617339</PMID><Article><Abstract>"
    b'<AbstractText Label="Background">'
    b"Human and <i>artificial</i> intelligence.</AbstractText>"
    b'<AbstractText Label="Conclusions">'
    b"They can work together.</AbstractText>"
    b"</Abstract></Article></MedlineCitation><PubmedData><ArticleIdList>"
    b'<ArticleId IdType="doi">10.1038/s41591-018-0300-7</ArticleId>'
    b"</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"
)


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


def test_missing_openalex_text_uses_matching_pubmed_abstract(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    pmids = []
    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(PUBMED_WORK),
            PUBMED_WORK,
            fetch=lambda url: pytest.fail("no PDF URL to fetch"),
            fetch_pubmed=lambda pmid: pmids.append(pmid) or PUBMED_XML,
        )
    )
    assert pmids == ["30617339"]
    assert document.coverage == "abstract_only"
    assert len(document.chunks) == 1
    assert document.chunks[0].kind == "abstract"
    assert document.chunks[0].text == (
        "Background: Human and artificial intelligence.\n\n"
        "Conclusions: They can work together."
    )
    assert document.chunks[0].locator["source"] == "pubmed"
    assert document.chunks[0].locator["doi"] == ("10.1038/s41591-018-0300-7")
    assert document.metadata["fulltext"]["status"] == "no_pdf_url"
    provenance = document.metadata["pubmed_abstract"]
    assert provenance["status"] == "parsed"
    assert provenance["pmid"] == "30617339"
    assert (
        provenance["sha256"] == document.chunks[0].locator["snapshot_sha256"]
    )
    assert (tmp_path / "raw" / provenance["sha256"]).read_bytes() == (
        PUBMED_XML
    )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (PUBMED_XML.replace(b"30617339", b"999"), "pmid_mismatch"),
        (
            PUBMED_XML.replace(b"10.1038/s41591-018-0300-7", b"10.1000/other"),
            "doi_mismatch",
        ),
        (
            PUBMED_XML.replace(
                b'<AbstractText Label="Background">'
                b"Human and <i>artificial</i> intelligence.</AbstractText>",
                b"",
            ).replace(
                b'<AbstractText Label="Conclusions">'
                b"They can work together.</AbstractText>",
                b"",
            ),
            "no_abstract",
        ),
        (b"<not-xml", "invalid_xml"),
        (b"<!ENTITY x 'bad'><PubmedArticleSet/>", "invalid_xml"),
        (b"<html>upstream error</html>", "invalid_response"),
    ],
)
def test_pubmed_abstract_rejects_unusable_records(raw, reason):
    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(PUBMED_WORK),
            PUBMED_WORK,
            fetch_pubmed=lambda pmid: raw,
        )
    )
    assert document.chunks == []
    assert document.coverage == "metadata_only"
    assert document.metadata["pubmed_abstract"]["status"] == reason


@pytest.mark.parametrize(
    ("work", "reason"),
    [
        (
            {**PUBMED_WORK, "ids": {"pmid": "https://evil.example/30617339"}},
            "missing_pmid",
        ),
        ({**PUBMED_WORK, "ids": {}}, "missing_pmid"),
        ({**PUBMED_WORK, "doi": None}, "missing_doi"),
    ],
)
def test_pubmed_fallback_requires_trusted_pmid_and_doi(work, reason):
    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(work),
            work,
            fetch_pubmed=lambda pmid: pytest.fail("must not fetch"),
        )
    )
    assert document.chunks == []
    assert document.metadata["pubmed_abstract"]["status"] == reason


def test_pubmed_fetch_failure_is_diagnostic():
    def unavailable(pmid):
        raise TimeoutError("temporary source outage")

    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(PUBMED_WORK),
            PUBMED_WORK,
            fetch_pubmed=unavailable,
        )
    )
    assert document.chunks == []
    assert document.metadata["pubmed_abstract"] == {
        "status": "fetch_failed",
        "pmid": "30617339",
        "doi": "10.1038/s41591-018-0300-7",
        "error": "TimeoutError",
    }


def test_failed_pdf_candidate_can_fall_back_to_pubmed(monkeypatch, tmp_path):
    monkeypatch.setenv("LCTREND_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(fulltext, "require_pdf_support", lambda: None)
    work = {
        **PUBMED_WORK,
        "best_oa_location": {
            "is_oa": True,
            "pdf_url": "https://example.org/unavailable.pdf",
        },
    }

    def no_pdf(url):
        raise ValueError("PDF unavailable")

    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(work),
            work,
            fetch=no_pdf,
            fetch_pubmed=lambda pmid: PUBMED_XML,
        )
    )
    assert document.metadata["fulltext"]["status"] == "failed"
    assert document.metadata["pubmed_abstract"]["status"] == "parsed"
    assert document.coverage == "abstract_only"


def test_pubmed_connector_uses_fixed_host_and_bounds_response(monkeypatch):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=PUBMED_XML)

    monkeypatch.setattr(connectors, "TRANSPORT", httpx.MockTransport(respond))
    assert asyncio.run(connectors.fetch_pubmed_xml("30617339")) == PUBMED_XML
    assert calls[0].url.host == "eutils.ncbi.nlm.nih.gov"
    assert calls[0].url.params["db"] == "pubmed"
    assert calls[0].url.params["id"] == "30617339"
    with pytest.raises(ValueError, match="numeric"):
        asyncio.run(connectors.fetch_pubmed_xml("30617339&db=other"))
    assert len(calls) == 1

    monkeypatch.setattr(
        connectors,
        "TRANSPORT",
        httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"x" * 1_000_001)
        ),
    )
    with pytest.raises(ValueError, match="size limit"):
        asyncio.run(connectors.fetch_pubmed_xml("30617339"))


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


def test_already_read_pdf_bytes_are_not_parsed_again(monkeypatch, tmp_path):
    # A-2: the PDF sha identifies the input of an earlier run.
    import hashlib

    monkeypatch.setattr(fulltext, "require_pdf_support", lambda: None)

    def parse(*args):
        pytest.fail("Known PDF bytes must not reach Docling")

    monkeypatch.setattr(fulltext, "_body_chunks", parse)
    raw = b"%PDF-1.7 fixture"
    document = asyncio.run(
        fulltext.attach_openalex_fulltext(
            parse_openalex(WORK),
            WORK,
            fetch=lambda url: raw,
            known_sha256=[hashlib.sha256(raw).hexdigest()],
        )
    )
    status = document.metadata["fulltext"]
    assert status["status"] == "already_processed"
    assert status["sha256"] == hashlib.sha256(raw).hexdigest()
    assert document.coverage == "abstract_only"


def test_prior_run_covers_only_the_input_it_read():
    from lctrend.ingest.processed import covers, run_input

    abstract_run = run_input({"input_coverage": "abstract_only"})
    pdf_run = run_input(
        '{"input_coverage": "abstract_and_full_text",'
        ' "input_fulltext_sha256": "aaa"}'
    )
    abstract = parse_openalex(WORK)
    assert covers([abstract_run], abstract)
    assert covers([pdf_run], abstract)
    with_pdf = parse_openalex(WORK)
    with_pdf.coverage = "abstract_and_full_text"
    with_pdf.metadata["fulltext"] = {"status": "parsed", "sha256": "aaa"}
    assert not covers([abstract_run], with_pdf)
    assert covers([pdf_run], with_pdf)
    with_pdf.metadata["fulltext"]["sha256"] = "bbb"
    assert not covers([pdf_run], with_pdf)
    # A run from before inputs were recorded is not paid for again.
    assert covers([run_input({})], with_pdf)


def test_pubmed_rate_limit_works_across_event_loops(monkeypatch):
    import asyncio

    from lctrend.ingest import connectors

    waits = []

    async def record_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(connectors.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(connectors, "_pubmed_next_slot", 0.0)
    # Each web job has its own loop; an asyncio.Lock would be bound to one.
    asyncio.run(connectors._pubmed_slot())
    asyncio.run(connectors._pubmed_slot())
    assert len(waits) == 2
    assert waits[1] > 0.3, "the second request waits for its slot"
