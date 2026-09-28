"""One node per company, full country names, organization types from the
catalog (feedback: "Intel (US)", "Samsung among organizations", "US")."""

from itertools import permutations

import pytest

from lctrend.core.models import ConceptKind, Mention
from lctrend.core.organizations import (
    company_key,
    country_names,
    country_suffix,
    display_rank,
    organization_identity,
    source_organization_type,
)
from lctrend.extraction.resolver import ConceptIndex, resolve_mentions
from lctrend.graph.normalize import plan_countries, plan_organizations
from lctrend.ingest.adapters import parse_openalex


def test_country_suffix_is_split_only_when_it_names_a_country():
    assert country_suffix("Intel (United States)") == ("Intel", "US")
    assert country_suffix("Microsoft Research (United Kingdom)") == (
        "Microsoft Research",
        "GB",
    )
    assert country_suffix("Institut Curie (Paris)") == (
        "Institut Curie (Paris)",
        None,
    )


@pytest.mark.parametrize(
    "name",
    ["Intel (United States)", "Intel (Germany)", "INTEL CORP", "intel"],
)
def test_every_spelling_of_a_company_is_one_identity(name):
    assert company_key(name) == "intel"
    assert (
        organization_identity(name, "company", "openalex", name)[0]
        == organization_identity("Intel", "company", "epo", "x")[0]
    )


def test_non_companies_keep_the_source_identity():
    # "Ministry of Health (Brazil)" and "(Japan)" are different bodies.
    first = organization_identity(
        "Ministry of Health (Brazil)", "government", "openalex", "I1"
    )
    second = organization_identity(
        "Ministry of Health (Japan)", "government", "openalex", "I2"
    )
    assert first[0] != second[0]
    assert first[1] == "Ministry of Health (Brazil)"


def test_a_role_is_not_a_type():
    assert source_organization_type("Samsung", "funder") == "company"
    assert source_organization_type("Université de Toulouse", "funder") == (
        "university"
    )
    assert source_organization_type("American Cancer Society", "funder") == (
        "funder"
    )
    # A source's own type is kept.
    assert source_organization_type("Cambridge Quantum", "company") == (
        "company"
    )


def test_mixed_case_short_name_is_the_display_name():
    names = ["INTEL CORP", "intel", "Intel"]
    assert min(names, key=display_rank) == "Intel"


def test_country_names_are_full_russian_and_english():
    assert country_names("DE") == ("Германия", "Germany")
    assert country_names("US") == ("Соединенные Штаты", "United States")


def test_openalex_country_copies_of_a_company_are_one_organization():
    document = parse_openalex(
        {
            "id": "https://openalex.org/W1",
            "title": "Edge inference",
            "publication_date": "2025-01-01",
            "authorships": [
                {
                    "author": {"id": "A1", "display_name": "Ann"},
                    "institutions": [
                        {
                            "id": "https://openalex.org/I1",
                            "display_name": "Intel (United States)",
                            "type": "company",
                            "country_code": "US",
                        }
                    ],
                },
                {
                    "author": {"id": "A2", "display_name": "Ben"},
                    "institutions": [
                        {
                            "id": "https://openalex.org/I2",
                            "display_name": "Intel (Germany)",
                            "type": "company",
                            "country_code": "DE",
                        }
                    ],
                },
            ],
            "grants": [
                {
                    "funder": "https://openalex.org/F1",
                    "funder_display_name": "Samsung",
                }
            ],
        }
    )
    organizations = {
        item.name: item.organization_type for item in document.organizations
    }
    assert organizations == {"Intel": "company", "Samsung": "company"}
    ann, ben = document.contributors
    assert ann.affiliation_ids == ben.affiliation_ids


def mention(index, text, kind):
    return Mention(
        mention_id=f"m{index}",
        chunk_id="c1",
        surface_text=text,
        canonical_text=text,
        start=0,
        end=len(text),
        type_candidates=[kind],
    )


def test_organization_kinds_are_one_identity_family():
    # "Acme Labs" typed differently by two documents is one organization,
    # of the most specific kind, in any order.
    names = [
        ("Acme Labs", ConceptKind.ORGANIZATION),
        ("Acme Labs", ConceptKind.COMPANY),
    ]
    results = set()
    for order in permutations(names):
        index = ConceptIndex()
        for number, (text, kind) in enumerate(order):
            resolve_mentions([mention(number, text, kind)], index)
        results.add(
            tuple((concept.concept_id, concept.kind) for concept in index)
        )
    assert len(results) == 1
    ((_, kind),) = results.pop()
    assert kind == ConceptKind.COMPANY


def test_normalization_plan_renames_countries_and_folds_companies():
    countries = plan_countries(
        [
            {"element_id": "1", "code": "US", "name": "US"},
            {"element_id": "2", "preferred_label": "JP", "name": "JP"},
            {
                "element_id": "3",
                "code": "DE",
                "name": "Германия",
                "name_en": "Germany",
            },
        ]
    )
    assert [(row["code"], row["name"]) for row in countries] == [
        ("US", "Соединенные Штаты"),
        ("JP", "Япония"),
    ]
    groups = plan_organizations(
        [
            {
                "organization_id": "organization:us",
                "name": "Intel (United States)",
                "organization_type": "company",
                "labels": ["Organization", "Company"],
            },
            {
                "organization_id": "organization:de",
                "name": "Intel (Germany)",
                "organization_type": "company",
                "labels": ["Organization", "Company"],
            },
            {
                "organization_id": "organization:samsung",
                "name": "Samsung",
                "organization_type": "funder",
                "labels": ["Organization"],
            },
            {
                "organization_id": "organization:acs",
                "name": "American Cancer Society",
                "organization_type": "funder",
                "labels": ["Organization"],
            },
        ]
    )
    plan = {group.name: group for group in groups}
    assert set(plan) == {"Intel", "Samsung"}
    assert plan["Intel"].members == ["organization:de", "organization:us"]
    assert plan["Samsung"].organization_type == "company"
