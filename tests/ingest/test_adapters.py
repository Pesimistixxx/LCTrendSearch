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


def test_openalex_maps_every_topic_to_a_canonical_domain():
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
        "Natural language processing",
        "Bioinformatics",
    ]
    assert document.domains[0].parent_name == "Artificial intelligence"


def test_openalex_normalizes_doi_identity_and_reads_ids_fallback():
    first = parse_openalex(
        {
            "id": "https://openalex.org/works/w123",
            "doi": "http://dx.doi.org/10.1234/ABC",
        }
    )
    second = parse_openalex(
        {
            "id": "W123",
            "ids": {
                "doi": "10.1234/abc",
                "pmid": "https://pubmed.ncbi.nlm.nih.gov/1",
            },
        }
    )
    assert first.document_id == second.document_id
    assert first.source.record_id == "W123"
    assert first.source.canonical_url == "https://doi.org/10.1234/abc"
    assert {item.scheme: item.value for item in second.identifiers} == {
        "openalex": "W123",
        "doi": "10.1234/abc",
        "pmid": "https://pubmed.ncbi.nlm.nih.gov/1",
    }


def test_openalex_current_reference_list_and_primary_topic_are_preserved():
    payload = {
        "id": "W123",
        "referenced_works": ["W1", "W2"],
        "primary_topic": {"display_name": "Named entity recognition"},
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A1",
                    "display_name": "Ada",
                    "orcid": "https://orcid.org/0000-0001-0002-0003",
                }
            }
        ],
    }
    document = parse_openalex(payload)
    assert document.metrics["reference_count"] == 2
    assert document.metadata["referenced_works"] == ["W1", "W2"]
    assert document.domains[0].name == "Natural language processing"
    assert document.contributors[0].external_ids[1].scheme == "orcid"
    payload["referenced_works_count"] = 0
    assert parse_openalex(payload).metrics["reference_count"] == 0


def test_openalex_abstract_ignores_invalid_positions_without_crashing():
    document = parse_openalex(
        {
            "id": "W123",
            "abstract_inverted_index": {
                "sensor": [2],
                "A": [0],
                "new": [1, None, "bad", False, -1],
                "missing": None,
            },
        }
    )
    assert document.chunks[0].text == "A new sensor"
    document = parse_openalex({"id": "W123", "abstract_inverted_index": []})
    assert document.coverage == "metadata_only"


def test_openalex_parses_current_awards_and_deduplicates_funders():
    award = {
        "id": "https://openalex.org/G1",
        "funder_award_id": "A-1",
        "funder_id": "https://openalex.org/F1",
        "funder_display_name": "Science Foundation",
    }
    document = parse_openalex(
        {
            "id": "W123",
            "awards": [award],
            "funders": [
                {
                    "id": "https://openalex.org/F1",
                    "display_name": "Science Foundation",
                }
            ],
        }
    )
    assert len(document.organizations) == 1
    assert document.organizations[0].role == "funder"
    assert document.metadata["awards"] == [award]
    assert document.metadata["grants"] == [
        {
            "funder": "Science Foundation",
            "award_id": "A-1",
        }
    ]


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
    # Counters are metric observations of the same content (A-1).
    payload["repository"]["stargazers_count"] = 100
    payload["repository"]["pushed_at"] = "2026-09-27"
    starred = parse_github(payload)
    assert starred.document_version_id == first_id
    assert starred.metrics["stars"] == 100
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


def test_openalex_rank_and_counters_do_not_make_a_new_version():
    work = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/x",
        "title": "Edge AI",
        "abstract_inverted_index": {"Edge": [0], "AI": [1]},
        "_retrieved_at": "2026-09-01T00:00:00Z",
    }
    first = parse_openalex(
        {**work, "relevance_score": 12.5, "cited_by_count": 3, "fwci": 1.1}
    )
    again = parse_openalex(
        {
            **work,
            "_retrieved_at": "2026-09-08T00:00:00Z",
            "relevance_score": 0.7,
            "cited_by_count": 9,
            "counts_by_year": [{"year": 2026, "cited_by_count": 6}],
            "updated_date": "2026-09-07T00:00:00",
            "fwci": 2.4,
            "citation_normalized_percentile": {"value": 0.9},
            "cited_by_percentile_year": {"min": 90, "max": 91},
        }
    )
    assert again.document_version_id == first.document_version_id
    assert again.metrics["citation_count"] == 9
    assert again.metrics["fwci"] == 2.4
    changed = parse_openalex(
        {**work, "abstract_inverted_index": {"Cloud": [0], "AI": [1]}}
    )
    assert changed.document_version_id != first.document_version_id


def test_pypi_new_release_of_another_line_keeps_the_version():
    package = {
        "info": {"name": "pkg", "version": "2.0", "summary": "Tool"},
        "releases": {"2.0": [{"upload_time": "2026-01-01"}]},
        "last_serial": 1,
    }
    backport = {
        **package,
        "releases": {**package["releases"], "1.9.1": []},
        "last_serial": 2,
    }
    assert (
        parse_pypi(package).document_version_id
        == parse_pypi(backport).document_version_id
    )
