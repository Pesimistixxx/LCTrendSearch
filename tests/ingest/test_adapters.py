import json

from lctrend.core.models import DocumentType
from lctrend.ingest.adapters import (
    parse_epo,
    parse_github,
    parse_openalex,
    parse_pypi,
)


def test_openalex_reconstructs_abstract_and_stable_identity():
    payload = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/example",
        "title": "A sensor",
        "language": "en",
        "publication_date": "2026-01-02",
        "abstract_inverted_index": {"A": [0], "new": [1], "sensor": [2]},
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Ada",
                },
                "countries": ["GB"],
                "institutions": [
                    {
                        "id": "https://openalex.org/I1",
                        "display_name": "Example University",
                        "type": "education",
                        "country_code": "GB",
                    }
                ],
            }
        ],
        "topics": [
            {
                "domain": {
                    "id": "https://openalex.org/domains/3",
                    "display_name": "Physical Sciences",
                }
            }
        ],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    first = parse_openalex(payload, raw)
    second = parse_openalex(payload, raw)
    assert first.document_type == DocumentType.ARTICLE
    assert first.document_id == second.document_id
    assert first.document_version_id == second.document_version_id
    assert first.chunks[0].text == "A new sensor"
    assert first.organizations[0].organization_type == "university"
    assert first.countries[0].code == "GB"
    assert first.domains == []
    assert first.contributors[0].affiliation_ids == [
        first.organizations[0].organization_id
    ]


def test_openalex_maps_one_canonical_domain_from_topics():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W2",
            "title": "NER for bioinformatics",
            "topics": [
                {
                    "display_name": "Named entity recognition",
                    "field": {"display_name": "Computer Science"},
                },
                {
                    "display_name": "Computational biology",
                    "field": {"display_name": "Life Sciences"},
                },
            ],
        }
    )
    assert [domain.name for domain in document.domains] == [
        "Natural language processing"
    ]
    assert document.domains[0].parent_name == "Artificial intelligence"


def test_github_decodes_readme_and_release():
    payload = {
        "repository": {
            "node_id": "R_1",
            "full_name": "org/repo",
            "name": "repo",
            "html_url": "https://github.com/org/repo",
            "pushed_at": "2026-01-01T00:00:00Z",
            "owner": {
                "login": "org",
                "type": "Organization",
                "node_id": "O_1",
            },
        },
        "readme": {"text": "# New method", "path": "README.md"},
        "releases": [{"id": 1, "tag_name": "v1", "body": "First release"}],
    }
    document = parse_github(payload)
    assert document.document_type == DocumentType.REPOSITORY
    assert [chunk.kind for chunk in document.chunks] == ["readme", "release"]
    assert document.organizations[0].name == "org"


def test_github_pins_commit_and_dates_each_content_kind():
    payload = {
        "repository": {
            "full_name": "org/repo",
            "created_at": "2000-01-01",
            "pushed_at": "2026-01-01",
        },
        "commit": {
            "sha": "abc123",
            "commit": {"committer": {"date": "2025-01-02T00:00:00Z"}},
        },
        "readme": {"text": "New technology"},
        "releases": [
            {
                "id": 1,
                "body": "New release",
                "published_at": "2026-02-03T00:00:00Z",
            }
        ],
        "_retrieved_at": "2026-09-26T00:00:00Z",
    }
    document = parse_github(payload)
    assert document.published_at == "2000-01-01"
    assert document.version_published_at == "2025-01-02T00:00:00Z"
    assert document.metrics_observed_at == payload["_retrieved_at"]
    assert document.chunks[0].locator["commit_sha"] == "abc123"
    assert (
        document.chunks[0].locator["observed_at"]
        == document.version_published_at
    )
    assert document.chunks[1].locator["observed_at"] == "2026-02-03T00:00:00Z"
    first_id = document.document_version_id
    payload["_retrieved_at"] = "2026-09-27T00:00:00Z"
    assert parse_github(payload).document_version_id == first_id
    payload["repository"]["stargazers_count"] = 100
    assert parse_github(payload).document_version_id != first_id
    first_id = parse_github(payload).document_version_id
    payload["readme"]["text"] = "Changed content"
    assert parse_github(payload).document_version_id != first_id


def test_legacy_github_timestamp_is_never_used_as_commit_sha_or_content_date():
    document = parse_github(
        {
            "repository": {
                "full_name": "org/repo",
                "created_at": "2000-01-01",
                "pushed_at": "2026-01-01",
            },
            "readme": {"text": "New"},
        }
    )
    assert document.version_published_at is None
    assert document.chunks[0].locator["commit_sha"] is None


def test_platform_is_not_an_independence_group_and_linked_package_shares_project():  # noqa: E501
    article = parse_openalex({"id": "https://openalex.org/W1"})
    package = parse_pypi({"info": {"name": "pkg"}})
    linked = parse_pypi(
        {
            "info": {
                "name": "pkg",
                "project_urls": {"Source": "https://github.com/ORG/repo.git"},
            }
        }
    )
    repository = parse_github({"full_name": "org/repo"})
    assert article.source.independence_group is None
    assert package.source.independence_group is None
    assert (
        linked.source.independence_group
        == repository.source.independence_group
    )


def test_pypi_extracts_description():
    document = parse_pypi(
        {
            "info": {
                "name": "example-package",
                "version": "1.2.3",
                "description": "Package documentation",
                "summary": "A model for NER",
                "country_code": "us",
                "author": "Ada, Grace",
                "maintainer": "Ada",
            },
            "urls": [{"upload_time_iso_8601": "2026-01-02T03:04:05Z"}],
        }
    )
    assert document.document_type == DocumentType.PACKAGE
    assert document.metadata["version"] == "1.2.3"
    assert document.published_at == "2026-01-02T03:04:05Z"
    assert [domain.name for domain in document.domains] == [
        "Natural language processing"
    ]
    assert [country.code for country in document.countries] == ["US"]
    assert document.chunks[0].text == "Package documentation"
    assert [person.name for person in document.contributors] == [
        "Ada",
        "Grace",
        "Ada",
    ]
    assert (
        document.contributors[0].contributor_id
        == document.contributors[2].contributor_id
    )


def test_pypi_chunks_markdown_by_section_and_size():
    document = parse_pypi(
        {
            "info": {
                "name": "example-package",
                "version": "1",
                "description": "# Intro\n\nShort.\n\n## Details\n\n"
                + "word " * 600,
            }
        }
    )
    assert len(document.chunks) > 2
    assert document.chunks[0].section_path == ["description", "Intro"]
    assert document.chunks[-1].section_path == [
        "description",
        "Intro",
        "Details",
    ]
    assert all(len(chunk.text) <= 1000 for chunk in document.chunks)
    assert any(chunk.locator["overlap_chars"] > 0 for chunk in document.chunks)


def test_epo_parses_namespaced_xml():
    xml = """
    <ops:world-patent-data xmlns:ops="http://ops.epo.org">
      <ops:publication-reference>
        <ops:country>EP</ops:country><ops:doc-number>123</ops:doc-number>
        <ops:kind>A1</ops:kind><ops:date>20260102</ops:date>
      </ops:publication-reference>
      <ops:invention-title>Energy efficient sensor</ops:invention-title>
      <ops:abstract><ops:p>A sensor using event-driven transmission.</ops:p></ops:abstract>
      <ops:applicant><ops:name>Example Labs</ops:name></ops:applicant>
      <ops:inventor><ops:name>Ada Lovelace</ops:name></ops:inventor>
    </ops:world-patent-data>
    """  # noqa: E501
    document = parse_epo(xml)
    assert document.document_type == DocumentType.PATENT
    assert document.title == "Energy efficient sensor"
    assert document.published_at == "2026-01-02"
    assert (
        document.chunks[0].text == "A sensor using event-driven transmission."
    )
