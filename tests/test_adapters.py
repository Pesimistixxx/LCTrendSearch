import json

from lctrend.adapters import parse_epo, parse_github, parse_openalex, parse_pypi
from lctrend.models import DocumentType


def test_openalex_reconstructs_abstract_and_stable_identity():
    payload = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1/example",
        "title": "A sensor",
        "language": "en",
        "publication_date": "2026-01-02",
        "abstract_inverted_index": {"A": [0], "new": [1], "sensor": [2]},
        "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Ada"}}],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    first = parse_openalex(payload, raw)
    second = parse_openalex(payload, raw)
    assert first.document_type == DocumentType.ARTICLE
    assert first.document_id == second.document_id
    assert first.document_version_id == second.document_version_id
    assert first.chunks[0].text == "A new sensor"


def test_github_decodes_readme_and_release():
    payload = {
        "repository": {
            "node_id": "R_1",
            "full_name": "org/repo",
            "name": "repo",
            "html_url": "https://github.com/org/repo",
            "pushed_at": "2026-01-01T00:00:00Z",
            "owner": {"login": "org", "type": "Organization", "node_id": "O_1"},
        },
        "readme": {"text": "# New method", "path": "README.md"},
        "releases": [{"id": 1, "tag_name": "v1", "body": "First release"}],
    }
    document = parse_github(payload)
    assert document.document_type == DocumentType.REPOSITORY
    assert [chunk.kind for chunk in document.chunks] == ["readme", "release"]
    assert document.contributors[0].kind == "organization"


def test_pypi_extracts_description():
    document = parse_pypi(
        {
            "info": {
                "name": "example-package",
                "version": "1.2.3",
                "description": "Package documentation",
                "author": "Ada",
            }
        }
    )
    assert document.document_type == DocumentType.PACKAGE
    assert document.metadata["version"] == "1.2.3"
    assert document.chunks[0].text == "Package documentation"


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
    """
    document = parse_epo(xml)
    assert document.document_type == DocumentType.PATENT
    assert document.title == "Energy efficient sensor"
    assert document.published_at == "2026-01-02"
    assert document.chunks[0].text == "A sensor using event-driven transmission."
