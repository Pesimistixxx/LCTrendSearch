"""Source pagination and material identity; all requests are mocked."""

import asyncio
import base64
from urllib.parse import parse_qs, urlsplit

import pytest

from lctrend.ingest.discovery import (
    discover_github,
    discover_openalex,
    discover_pypi_from_github_payload,
    material_identity,
)


@pytest.mark.parametrize(
    "doi",
    [
        "10.1234/AbC",
        " DOI: 10.1234/AbC ",
        "https://doi.org/10.1234/AbC",
        "http://dx.doi.org/10.1234/AbC",
        "https://DOI.ORG/10.1234%2FAbC?download=1#section",
    ],
)
def test_doi_identity_is_shared_across_forms_and_sources(doi):
    assert (
        material_identity("openalex", "W123", {"doi": doi})
        == "doi:10.1234/abc"
    )
    assert (
        material_identity("crossref", "different-id", {"ids": {"doi": doi}})
        == "doi:10.1234/abc"
    )


def test_invalid_doi_link_is_not_a_cross_source_identity():
    assert (
        material_identity(
            "openalex",
            "https://openalex.org/W123",
            {"doi": "https://example.org/10.1234/abc"},
        )
        == "openalex:w123"
    )
    assert (
        material_identity("openalex", "W123", {"doi": "not a doi"})
        == "openalex:w123"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "Org/Repo",
        "https://github.com/ORG/REPO/",
        "https://github.com/Org/Repo.git",
        "https://api.github.com/repos/Org/Repo",
    ],
)
def test_github_case_and_url_aliases_have_one_material_id(identifier):
    assert material_identity("github", identifier) == "github:org/repo"


def test_hydrated_and_discovery_github_payload_have_same_identity():
    plain = {"full_name": "Org/Repo", "description": "Study"}
    fetched = {
        "repository": plain,
        "readme": {"text": "Different content"},
        "commit": {"sha": "123"},
    }
    assert material_identity("github", "Org/Repo", plain) == material_identity(
        "github", "Org/Repo", fetched
    )


def test_repository_package_and_accompanying_paper_are_distinct_materials():
    doi = "https://doi.org/10.1234/accompanying-paper"
    paper = material_identity("openalex", "W123", {"doi": doi})
    repository = material_identity(
        "github", "Org/Repo", {"full_name": "Org/Repo", "doi": doi}
    )
    package = material_identity(
        "pypi", "Sensor_Kit", {"info": {"name": "Sensor_Kit"}, "doi": doi}
    )
    assert paper == "doi:10.1234/accompanying-paper"
    assert repository == "github:org/repo"
    assert package == "pypi:sensor-kit"
    assert len({paper, repository, package}) == 3


def test_unknown_source_doi_metadata_does_not_imply_article_identity():
    assert (
        material_identity(
            "website",
            "https://example.org/material",
            {"doi": "10.1234/reference"},
        )
        == "website:https://example.org/material"
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "My_Package",
        "my.package",
        "my---package",
        "https://pypi.org/project/My_Package/1.0/",
    ],
)
def test_pypi_pep503_name_variants_have_one_identity(identifier):
    assert material_identity("pypi", identifier) == "pypi:my-package"


def test_discovery_preserves_openalex_full_record_and_real_total(monkeypatch):
    calls = []
    work = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1234/One",
        "title": "Paper",
        "abstract_inverted_index": {"Sensor": [0]},
    }

    def fetch(*args):
        calls.append(args)
        return {
            "results": [work],
            "meta": {"count": 13000, "next_cursor": "next"},
        }

    monkeypatch.setenv("OPENALEX_MAILTO", "research@example.test")
    monkeypatch.setattr("lctrend.ingest.discovery.fetch_openalex_page", fetch)
    page = asyncio.run(discover_openalex(" edge computing ", "cursor"))
    assert calls == [
        ("edge computing", "cursor", 100, "research@example.test", None)
    ]
    assert page["total"] == 13000
    assert page["next_cursor"] == "next"
    assert page["complete"] is False
    assert page["limitations"] == []
    assert page["items"][0]["canonical_id"] == "doi:10.1234/one"
    assert page["items"][0]["payload"] == work


def test_openalex_terminal_page_is_complete(monkeypatch):
    monkeypatch.setattr(
        "lctrend.ingest.discovery.fetch_openalex_page",
        lambda *args: {
            "results": [],
            "meta": {"count": 0, "next_cursor": None},
        },
    )
    page = asyncio.run(discover_openalex("quantum"))
    assert page == {
        "items": [],
        "next_cursor": None,
        "total": 0,
        "complete": True,
        "limitations": [],
    }


def test_openalex_empty_stale_cursor_reports_limited_coverage(monkeypatch):
    monkeypatch.setattr(
        "lctrend.ingest.discovery.fetch_openalex_page",
        lambda *args: {
            "results": [],
            "meta": {"count": 100, "next_cursor": "stale"},
        },
    )
    page = asyncio.run(discover_openalex("quantum"))
    assert page["next_cursor"] is None
    assert page["complete"] is False
    assert page["limitations"][0]["code"] == "empty_page_with_cursor"


def repo(name="Org/Repo", **overrides):
    return {
        "full_name": name,
        "html_url": "https://github.com/" + name,
        **overrides,
    }


def test_github_query_and_token_are_only_sent_to_repository_search(
    monkeypatch,
):
    calls = []

    def fetch(url, headers):
        calls.append((url, headers))
        return {
            "total_count": 1,
            "incomplete_results": False,
            "items": [repo(fork=True, stargazers_count=0)],
        }

    monkeypatch.setenv("GITHUB_TOKEN", "test-secret")
    monkeypatch.setattr("lctrend.ingest.discovery.fetch_json", fetch)
    page = asyncio.run(discover_github("edge + robots"))
    url, headers = calls[0]
    assert urlsplit(url).path == "/search/repositories"
    assert parse_qs(urlsplit(url).query) == {
        "q": ["edge + robots"],
        "per_page": ["100"],
        "page": ["1"],
    }
    assert headers["Authorization"] == "Bearer test-secret"
    assert page["complete"] is True
    assert (
        page["items"][0]["payload"]["fork"] is True
    )  # No candidate filtering.
    assert page["items"][0]["source_id"] == "org/repo"
    assert "test-secret" not in str(page)


def test_github_next_page_and_final_cap_never_claim_entire_source_complete(
    monkeypatch,
):
    monkeypatch.setattr(
        "lctrend.ingest.discovery.fetch_json",
        lambda *args: {
            "items": [repo()],
            "total_count": 12345,
            "incomplete_results": False,
        },
    )
    first = asyncio.run(discover_github("robotics"))
    assert first["next_cursor"] == "2"
    assert first["total"] == 1000
    assert first["complete"] is False
    assert first["limitations"][0]["reported_total"] == 12345
    last = asyncio.run(discover_github("robotics", "10"))
    assert last["next_cursor"] is None
    assert last["complete"] is False
    assert last["limitations"][0]["code"] == "github_search_cap"


def test_github_under_cap_final_page_is_complete(monkeypatch):
    monkeypatch.setattr(
        "lctrend.ingest.discovery.fetch_json",
        lambda *args: {
            "items": [repo()],
            "total_count": 101,
            "incomplete_results": False,
        },
    )
    assert asyncio.run(discover_github("robotics", "1"))["next_cursor"] == "2"
    last = asyncio.run(discover_github("robotics", "2"))
    assert last["next_cursor"] is None
    assert last["complete"] is True


def test_github_incomplete_flag_is_visible_even_under_cap(monkeypatch):
    monkeypatch.setattr(
        "lctrend.ingest.discovery.fetch_json",
        lambda *args: {
            "items": [repo()],
            "total_count": 1,
            "incomplete_results": True,
        },
    )
    page = asyncio.run(discover_github("robotics"))
    assert page["next_cursor"] is None
    assert page["complete"] is False
    assert page["limitations"][0]["code"] == "github_incomplete_results"


@pytest.mark.parametrize("cursor", ["0", "11", "garbage"])
def test_invalid_github_page_does_not_request_source(monkeypatch, cursor):
    calls = []
    monkeypatch.setattr(
        "lctrend.ingest.discovery.fetch_json", lambda *args: calls.append(args)
    )
    with pytest.raises(ValueError):
        asyncio.run(discover_github("robotics", cursor))
    assert calls == []


@pytest.mark.parametrize("discover", [discover_openalex, discover_github])
def test_empty_topic_is_not_silently_replaced_with_an_unrelated_query(
    discover,
):
    with pytest.raises(ValueError):
        asyncio.run(discover(" "))


def test_readme_links_deduplicate_variants_without_inferred_names():
    text = """SensorMagic does not imply a PyPI package.
    [package](https://pypi.org/project/My_Package/) and https://pypi.org/project/my.package/1.0/
    Ignore https://pypi.org.evil.test/project/false and https://other.test/project/false
    A second package: https://pypi.org/project/another-package/
    """
    refs = discover_pypi_from_github_payload({"readme": {"text": text}})
    assert [r["source_id"] for r in refs] == ["my-package", "another-package"]
    assert all(r["payload"] is None and r["source"] == "pypi" for r in refs)
    assert refs[0]["canonical_id"] == "pypi:my-package"


def test_fetch_github_base64_readme_is_supported():
    text = "Install [sensor](https://pypi.org/project/Sensor_Kit/)."
    encoded = base64.b64encode(text.encode()).decode()
    refs = discover_pypi_from_github_payload(
        {
            "repository": repo(),
            "readme": {"content": encoded, "encoding": "base64"},
        }
    )
    assert [ref["source_id"] for ref in refs] == ["sensor-kit"]


def test_bare_pypi_link_at_sentence_end_is_not_lost():
    refs = discover_pypi_from_github_payload(
        {
            "readme": {
                "text": "Published at https://pypi.org/project/sensor-kit."
            }
        }
    )
    assert [ref["source_id"] for ref in refs] == ["sensor-kit"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"readme": None},
        {"readme": {"content": "!broken"}},
        {"readme": {"text": "pip install guessed-name"}},
    ],
)
def test_no_readme_link_means_no_package_discovery(payload):
    assert discover_pypi_from_github_payload(payload) == []
