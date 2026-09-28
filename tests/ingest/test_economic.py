"""The economic layer: grants and vacancies with structured money, the
same organizations as the rest of the corpus, no personal contacts."""

import asyncio

import pytest

from lctrend.core.models import DocumentType
from lctrend.core.organizations import organization_identity
from lctrend.ingest import economic
from lctrend.ingest.economic import (
    discover_hh,
    discover_trudvsem,
    parse_hh,
    parse_nih,
    parse_nsf,
    parse_trudvsem,
)

NIH = {
    "appl_id": 11456661,
    "project_num": "1R61HD123541-01",
    "fiscal_year": 2024,
    "award_amount": 493414,
    "award_notice_date": "2024-09-17T00:00:00",
    "project_start_date": "2024-09-21T00:00:00",
    "project_end_date": "2026-08-31T00:00:00",
    "project_title": "Retrieval-augmented generation for clinical notes",
    "abstract_text": "We build retrieval-augmented generation for notes.",
    "phr_text": "Better notes help patients.",
    "organization": {
        "org_name": "INTEL CORPORATION",
        "org_country": "UNITED STATES",
        "external_org_id": 99,
    },
    "organization_type": {"name": "DOMESTIC FOR-PROFITS"},
    "agency_ic_admin": {
        "code": "LM",
        "abbreviation": "NLM",
        "name": "National Library of Medicine",
    },
    "principal_investigators": [
        {"profile_id": 1, "full_name": "Ann  Lee", "is_contact_pi": True}
    ],
    "project_detail_url": "https://reporter.nih.gov/project-details/1",
}

NSF = {
    "id": "2555329",
    "title": "Securing Retrieval-Augmented Generation of LLMs",
    "abstractText": "Retrieval-augmented generation augments LLMs.",
    "fundsObligatedAmt": "561791",
    "estimatedTotalAmt": "600000",
    "awardeeName": "Pennsylvania State Univ University Park",
    "awardeeCountryCode": "US",
    "ueiNumber": "NPM2J7MSCF61",
    "date": "08/04/2025",
    "startDate": "10/01/2025",
    "expDate": "09/30/2028",
    "piFirstName": "Jinyuan",
    "piLastName": "Jia",
}

TRUDVSEM = {
    "id": "6ddd5624",
    "job-name": "Инженер по машинному обучению",
    "creation-date": "2025-07-03",
    "salary_min": 200000,
    "salary_max": 300000,
    "currency": "«руб.»",
    "company": {"name": "ПАО Сбербанк", "inn": "7707083893"},
    "region": {"name": "Город Москва"},
    "duty": "Обучать модели машинного обучения для кредитного скоринга.",
    "requirement": {"education": "Высшее", "experience": 3},
    "vac_url": "https://trudvsem.ru/vacancy/card/x/6ddd5624",
}

HH = {
    "id": "101",
    "name": "ML-инженер (RAG)",
    "published_at": "2025-03-01T10:00:00+0300",
    "salary": {"from": 250000, "to": None, "currency": "RUR", "gross": False},
    "employer": {"id": "3529", "name": "Сбер"},
    "description": "<p>Строим <b>retrieval-augmented generation</b>.</p>",
    "key_skills": [{"name": "LangChain"}, {"name": "PyTorch"}],
    "alternate_url": "https://hh.ru/vacancy/101",
}


def test_nih_award_is_a_grant_with_constant_dollar_money():
    document = parse_nih(NIH)
    assert document.document_type == DocumentType.GRANT
    assert document.published_at == "2024-09-17"
    (fact,) = document.economic_facts
    assert (fact.category, fact.amount, fact.currency, fact.period) == (
        "grant_award",
        493414,
        "USD",
        "per_year",
    )
    assert fact.amount_usd_real == pytest.approx(
        493414 * 321.962 / 313.698, rel=1e-6
    )
    recipient, funder = document.organizations
    # The recipient is the same Intel as in OpenAlex and patents.
    assert (
        recipient.organization_id
        == organization_identity(
            "Intel (United States)", "company", "openalex", "I1"
        )[0]
    )
    assert fact.recipient_organization_id == recipient.organization_id
    assert fact.payer_organization_id == funder.organization_id
    assert funder.role == "funder"
    assert [country.code for country in document.countries] == ["US"]
    assert [chunk.kind for chunk in document.chunks] == [
        "title",
        "abstract",
        "public_health_relevance",
    ]
    assert document.contributors[0].name == "Ann Lee"


def test_nsf_award_keeps_obligated_and_estimated_amounts():
    document = parse_nsf(NSF)
    (fact,) = document.economic_facts
    assert (fact.amount, fact.amount_max) == (561791, 600000)
    assert fact.observed_at == "2025-08-04"
    assert fact.real_status == "exact"
    assert document.organizations[0].organization_type == "university"


def test_vacancies_carry_monthly_salary_ranges_of_the_employer():
    document = parse_trudvsem({"vacancy": TRUDVSEM})
    assert document.document_type == DocumentType.JOB_POSTING
    (fact,) = document.economic_facts
    assert (fact.amount, fact.amount_max, fact.currency, fact.period) == (
        200000,
        300000,
        "RUB",
        "per_month",
    )
    employer = document.organizations[0]
    assert employer.organization_type == "company"
    assert fact.payer_organization_id == employer.organization_id

    card = parse_hh(HH)
    (offer,) = card.economic_facts
    # hh.ru writes rubles as RUR; an open range has no upper bound.
    assert (offer.amount, offer.amount_max, offer.currency) == (
        250000,
        None,
        "RUB",
    )
    assert "retrieval-augmented generation" in card.chunks[1].text
    assert card.metadata["skills"] == ["LangChain", "PyTorch"]


def test_contacts_of_people_are_never_stored(monkeypatch):
    async def fake(url, headers=None, json_body=None):
        return {
            "meta": {"total": 1},
            "results": {
                "vacancies": [
                    {
                        "vacancy": {
                            **TRUDVSEM,
                            "contact_person": "Иван Петров",
                            "contact_list": [{"contact_value": "+7 900"}],
                        }
                    }
                ]
            },
        }

    monkeypatch.setattr(economic, "fetch_json", fake)
    page = asyncio.run(discover_trudvsem("машинное обучение"))
    (item,) = page["items"]
    assert "contact_person" not in item["payload"]
    assert "contact_list" not in item["payload"]


def test_loose_vacancy_matches_are_dropped(monkeypatch):
    # trudvsem finds "машинист ... машины (с обучением)" for the phrase.
    unrelated = {
        **TRUDVSEM,
        "id": "other",
        "job-name": "Машинист железнодорожно-строительной машины",
        "duty": "Работа на машине с обучением на месте.",
    }

    async def fake(url, headers=None, json_body=None):
        return {
            "meta": {"total": 2},
            "results": {
                "vacancies": [{"vacancy": TRUDVSEM}, {"vacancy": unrelated}]
            },
        }

    monkeypatch.setattr(economic, "fetch_json", fake)
    page = asyncio.run(discover_trudvsem("машинное обучение"))
    assert [item["source_id"] for item in page["items"]] == ["6ddd5624"]


def test_hh_without_token_reports_the_limitation(monkeypatch):
    monkeypatch.delenv("HH_ACCESS_TOKEN", raising=False)
    page = asyncio.run(discover_hh("python"))
    assert page["items"] == []
    assert page["limitations"][0]["code"] == "hh_token_missing"
