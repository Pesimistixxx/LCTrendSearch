"""Document metadata needed by trend features: dates, parties, domains."""

import json

from lctrend.ingest.adapters import parse_epo, parse_github, parse_openalex
from lctrend.ingest.file_adapters import parse_file

EPO = """
<ops:world-patent-data xmlns:ops="http://ops.epo.org">
  <ops:priority-claim><ops:date>20190101</ops:date></ops:priority-claim>
  <ops:publication-reference>
    <ops:document-id>
      <ops:country>EP</ops:country><ops:doc-number>123</ops:doc-number>
      <ops:kind>A1</ops:kind><ops:date>20260102</ops:date>
    </ops:document-id>
  </ops:publication-reference>
  <ops:invention-title lang="en">Robotic gripper</ops:invention-title>
  <ops:abstract lang="en"><ops:p>A robotic gripper for sorting.</ops:p></ops:abstract>
  <ops:applicant data-format="epodoc"><ops:name>ACME ROBOTICS GMBH [DE]</ops:name></ops:applicant>
  <ops:applicant data-format="original"><ops:name>Acme Robotics GmbH</ops:name></ops:applicant>
  <ops:applicant data-format="epodoc"><ops:name>TECHNISCHE UNIVERSITAET MUENCHEN [DE]</ops:name></ops:applicant>
  <ops:applicant data-format="epodoc"><ops:name>LOVELACE ADA [GB]</ops:name></ops:applicant>
  <ops:inventor data-format="epodoc"><ops:name>LOVELACE ADA [GB]</ops:name></ops:inventor>
  <ops:claims><ops:claim><ops:claim-text>1. A gripper comprising a sensor.</ops:claim-text></ops:claim></ops:claims>
</ops:world-patent-data>
"""  # noqa: E501


def test_epo_types_applicants_keeps_residence_and_publication_date():
    document = parse_epo(EPO, retrieved_at="2026-09-27T00:00:00Z")
    assert document.published_at == "2026-01-02"
    assert document.language == "en"
    assert document.retrieved_at == "2026-09-27T00:00:00Z"
    organizations = {
        item.name: (item.organization_type, item.country_code)
        for item in document.organizations
    }
    # One epodoc record per applicant; the original-format duplicate is not a
    # second company.
    assert organizations == {
        "ACME ROBOTICS GMBH": ("company", "DE"),
        "TECHNISCHE UNIVERSITAET MUENCHEN": ("university", "DE"),
    }
    # An inventor filing in their own name is a person.
    assert ("LOVELACE ADA", "applicant") in {
        (item.name, item.role) for item in document.contributors
    }
    # A-7: EP is the European Patent Office, not a country; the old
    # expectation {"EP", "DE"} made it a jurisdiction country.
    assert {item.code for item in document.countries} == {"DE"}
    assert document.metadata["patent_office"] == "EP"
    assert [item.name for item in document.domains] == ["Robotics"]
    assert [chunk.kind for chunk in document.chunks] == ["abstract", "claims"]
    assert document.coverage == "full_text"


def test_github_repository_gets_domains_and_typed_owner():
    document = parse_github(
        {
            "repository": {
                "full_name": "acme-inc/vision",
                "name": "vision",
                "description": "Computer vision toolkit for robotics",
                "topics": ["deep-learning"],
                "owner": {"login": "Acme Inc", "type": "Organization"},
            }
        }
    )
    assert {item.name for item in document.domains} == {
        "Computer vision",
        "Robotics",
    }
    assert document.organizations[0].organization_type == "company"


def test_openalex_keeps_funders_venue_and_uncatalogued_subfields():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W9",
            "title": "Soil sensing",
            "publication_date": "2025-05-05",
            "grants": [
                {
                    "funder": "https://openalex.org/F1",
                    "funder_display_name": "Example Science Foundation",
                    "award_id": "A-1",
                }
            ],
            "primary_location": {
                "source": {
                    "display_name": "Sensors Journal",
                    "type": "journal",
                }
            },
            "topics": [
                {
                    "display_name": "Soil moisture sensing",
                    "subfield": {
                        "id": "https://openalex.org/subfields/1111",
                        "display_name": "Soil Science",
                    },
                    "field": {"display_name": "Agricultural Sciences"},
                }
            ],
        }
    )
    funder = document.organizations[0]
    assert (funder.name, funder.role) == (
        "Example Science Foundation",
        "funder",
    )
    assert document.metadata["grants"] == [
        {"funder": "Example Science Foundation", "award_id": "A-1"}
    ]
    assert document.metadata["venue"] == "Sensors Journal"
    assert [(d.name, d.parent_name) for d in document.domains] == [
        ("Soil Science", "Agricultural Sciences")
    ]


def test_local_file_sidecar_supplies_dates_parties_and_source(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("Отчёт о развитии роботов.", encoding="utf-8")
    (tmp_path / "report.txt.meta.json").write_text(
        json.dumps(
            {
                "published_at": "2025-11-20",
                "document_type": "report",
                "source": {
                    "name": "Ministry report",
                    "type": "government",
                    "family": "regulatory",
                    "reliability_tier": 3,
                },
                "authors": [{"name": "Ada", "affiliations": ["Acme Ltd"]}],
                "organizations": [
                    {"name": "Acme Ltd", "type": "company", "country": "gb"}
                ],
                "countries": ["RU"],
                "domains": ["Robotics"],
            }
        ),
        encoding="utf-8",
    )
    document = parse_file(path)
    assert document.published_at == "2025-11-20"
    assert document.metadata["metadata_basis"]["published_at"] == "sidecar"
    assert document.language == "ru"
    assert document.source.source_family == "regulatory"
    assert document.organizations[0].country_code == "GB"
    assert document.contributors[0].affiliation_ids == [
        document.organizations[0].organization_id
    ]
    assert {item.code for item in document.countries} == {"GB", "RU"}
    assert [item.name for item in document.domains] == ["Robotics"]


def test_html_publication_meta_is_a_date_but_file_creation_is_not(tmp_path):
    page = tmp_path / "news.html"
    page.write_text(
        '<html lang="en-US"><head>'
        '<meta name="citation_publication_date" content="2024/06/03">'
        '<meta name="citation_author" content="Grace Hopper">'
        "</head><body><p>The new chip and the old one.</p></body></html>",
        encoding="utf-8",
    )
    document = parse_file(page)
    assert document.published_at == "2024-06-03"
    assert document.language == "en"
    assert [item.name for item in document.contributors] == ["Grace Hopper"]
    plain = tmp_path / "notes.txt"
    plain.write_text("Undated notes about the chip and the board.", "utf-8")
    undated = parse_file(plain)
    assert undated.published_at is None
    assert undated.metadata["metadata_basis"]["language"] == "script_heuristic"


def test_patent_offices_are_not_countries():
    # A-7: WO/EP/EA publications became "countries" of the document.
    for office in ("WO", "EA"):
        document = parse_epo(
            EPO.replace(
                "<ops:country>EP</ops:country>",
                f"<ops:country>{office}</ops:country>",
            ).replace("[DE]", f"[{office}]", 1),
            retrieved_at="2026-09-27T00:00:00Z",
        )
        assert office not in {item.code for item in document.countries}
        assert document.metadata["patent_office"] == office
        assert all(
            item.country_code != office for item in document.organizations
        )
