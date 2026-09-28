"""The economic layer: sources whose records carry money and demand.

Grants (NIH RePORTER, NSF Awards) say how much money goes into the work a
technology names; vacancies (Работа России / trudvsem, hh.ru) say who
hires for it and at what salary. They run in parallel with the scholarly
and code sources: the same crawl directions, the same extraction of
technologies from their text, the same organizations.

What differs is money. It is a structured field of the record, never an
LLM reading, so every record also yields :class:`EconomicFact` rows with
the nominal amount, its currency and date, and its value in constant
dollars (core.money). Organizations are identified the way the other
adapters identify them (core.organizations): a company funded by NSF,
publishing on OpenAlex and hiring on hh.ru is one node, which is what
makes money, papers and demand comparable per organization.

Contact data of people (phones, e-mails, recruiters) is dropped before a
record is stored.
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode

from ..core.aio import resolve
from ..core.config import load_catalog
from ..core.models import (
    Contributor,
    Country,
    DocumentEnvelope,
    DocumentType,
    EconomicFact,
    ExternalId,
    Organization,
    stable_id,
)
from ..core.money import real_fields
from ..core.organizations import (
    country_code_of,
    organization_identity,
    source_organization_type,
)
from ..extraction.lexical import lexical_tokens
from .adapters import (
    _artifact,
    _domains_from_values,
    _markdown_chunks,
    _observation_hash,
    _source,
)
from .connectors import fetch_json

ECONOMIC_SOURCES = ("nih", "nsf", "trudvsem", "hh")

# Fields that identify or reach a person privately; never stored.
_PRIVATE = frozenset(
    {
        "contact_list",
        "contact_person",
        "contacts",
        "piEmail",
        "poEmail",
        "poPhone",
        "awardeePhone",
        "coPDPI",
        "pi",
        "program_officers",
    }
)


def _settings(platform: str) -> Mapping[str, Any]:
    return load_catalog("sources")["platforms"][platform]


def _without_private(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _without_private(item)
            for key, item in value.items()
            if key not in _PRIVATE
        }
    if isinstance(value, list):
        return [_without_private(item) for item in value]
    return value


def _number(value: Any) -> Optional[float]:
    try:
        number = float(str(value).replace(" ", "").replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _iso_date(value: Any) -> Optional[str]:
    """ISO date of "2026-09-17T00:00:00", "08/04/2026" or "2026-08-04"."""
    if not value:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", text)
    if match:
        month, day, year = match.groups()
        return f"{year}-{month}-{day}"
    match = re.match(r"\d{4}-\d{2}-\d{2}", text)
    return match.group() if match else None


def _year(value: Optional[str]) -> Optional[int]:
    return int(value[:4]) if value and re.match(r"\d{4}", value) else None


def _text(value: Any) -> str:
    """Plain text of an HTML fragment (hh.ru descriptions)."""
    text = re.sub(r"<\s*(?:br|/p|/li|/div)\s*/?>", "\n", str(value or ""))
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return "\n".join(
        " ".join(line.split()) for line in text.splitlines() if line.strip()
    )


def _fact(
    document_id: str,
    category: str,
    amount: Optional[float],
    currency: Optional[str],
    observed_at: Optional[str],
    *,
    amount_max: Optional[float] = None,
    period: str = "total",
    year: Optional[int] = None,
    recipient: Optional[str] = None,
    payer: Optional[str] = None,
    source_field: Optional[str] = None,
) -> EconomicFact:
    year = year or _year(observed_at)
    upper = real_fields(amount_max, currency, year)["amount_usd_real"]
    return EconomicFact(
        fact_id=stable_id(
            "economic_fact", document_id, category, source_field
        ),
        category=category,
        amount=amount,
        amount_max=amount_max,
        currency=currency,
        period=period,
        observed_at=observed_at,
        recipient_organization_id=recipient,
        payer_organization_id=payer,
        source_field=source_field,
        amount_max_usd_real=upper,
        **real_fields(amount, currency, year),
    )


def _organization(
    name: str,
    source: str,
    external_id: str,
    source_type: Optional[str],
    role: str,
    country: Optional[str] = None,
    scheme: Optional[str] = None,
) -> Organization:
    kind = source_organization_type(name, source_type)
    organization_id, display = organization_identity(
        name, kind, source, external_id
    )
    return Organization(
        organization_id=organization_id,
        name=display,
        organization_type=kind,
        country_code=country,
        role=role,
        external_ids=[ExternalId(scheme=scheme or source, value=external_id)],
    )


def _countries(*codes: Optional[str]) -> List[Country]:
    return [
        Country(
            country_id=stable_id("country", code), code=code, role="metadata"
        )
        for code in dict.fromkeys(code for code in codes if code)
    ]


def _chunks(version_id: str, sections: List[tuple]) -> list:
    chunks = []
    for kind, text in sections:
        if text and text.strip():
            chunks.extend(
                _markdown_chunks(version_id, kind, text.strip(), len(chunks))
            )
    return chunks


def _document(
    platform: str,
    record_id: str,
    payload: Mapping[str, Any],
    raw: Optional[bytes],
    **fields: Any,
) -> DocumentEnvelope:
    raw = raw or json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    url = fields.pop("url")
    document_id = fields.pop("document_id")
    version_id = fields.pop("version_id")
    chunks = fields.pop("chunks")
    return DocumentEnvelope(
        document_id=document_id,
        document_version_id=version_id,
        retrieved_at=payload.get("_retrieved_at"),
        source=_source(platform, record_id, url),
        artifact=_artifact(url, raw, "application/json"),
        identifiers=[ExternalId(scheme=platform, value=record_id)],
        chunks=chunks,
        coverage="full_text" if chunks else "metadata_only",
        **fields,
    )


# ------------------------------------------------------------------ grants


def _nih_organization_type(payload: Mapping[str, Any]) -> Optional[str]:
    kind = str((payload.get("organization_type") or {}).get("name") or "")
    kind = kind.upper()
    if "FOR-PROFIT" in kind or "SMALL BUSINESS" in kind:
        return "company"
    if any(word in kind for word in ("SCHOOL", "UNIVERSIT", "HIGHER ED")):
        return "university"
    if "HOSPITAL" in kind:
        return "healthcare"
    return None


def parse_nih(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    """One NIH award (one fiscal year of a project) as a grant document."""
    application = str(payload["appl_id"])
    document_id = stable_id("document", "nih", application)
    version_id = stable_id(
        "version", document_id, _observation_hash(payload, "nih")
    )
    organization = payload.get("organization") or {}
    country = country_code_of(organization.get("org_country")) or (
        organization.get("org_fips")
        if re.fullmatch(r"[A-Z]{2}", str(organization.get("org_fips") or ""))
        else None
    )
    organizations = []
    recipient = None
    if organization.get("org_name"):
        recipient = _organization(
            organization["org_name"],
            "nih",
            str(
                organization.get("external_org_id") or organization["org_name"]
            ),
            _nih_organization_type(payload),
            "grant_recipient",
            country,
        )
        organizations.append(recipient)
    agency = payload.get("agency_ic_admin") or {}
    payer = None
    if agency.get("name"):
        payer = _organization(
            agency["name"],
            "nih",
            "ic:" + str(agency.get("code") or agency["name"]),
            "government",
            "funder",
            "US",
        )
        organizations.append(payer)
    announced = _iso_date(
        payload.get("award_notice_date")
        or payload.get("budget_start")
        or payload.get("project_start_date")
    )
    fiscal_year = payload.get("fiscal_year")
    fact = _fact(
        document_id,
        "grant_award",
        _number(payload.get("award_amount")),
        "USD",
        announced,
        # A fiscal year's award, valued in that year's dollars.
        year=int(fiscal_year) if fiscal_year else None,
        period="per_year",
        recipient=recipient.organization_id if recipient else None,
        payer=payer.organization_id if payer else None,
        source_field="award_amount",
    )
    title = str(payload.get("project_title") or application)
    chunks = _chunks(
        version_id,
        [
            ("title", title),
            ("abstract", payload.get("abstract_text")),
            ("public_health_relevance", payload.get("phr_text")),
        ],
    )
    contributors = [
        Contributor(
            contributor_id=stable_id(
                "person", "nih", str(person.get("profile_id") or name)
            ),
            name=name,
            role="principal_investigator",
        )
        for person in payload.get("principal_investigators") or []
        if (name := " ".join(str(person.get("full_name") or "").split()))
    ]
    return _document(
        "nih",
        application,
        payload,
        raw,
        url=payload.get("project_detail_url")
        or f"https://reporter.nih.gov/project-details/{application}",
        document_id=document_id,
        version_id=version_id,
        chunks=chunks,
        document_type=DocumentType.GRANT,
        title=title,
        language="en",
        published_at=announced,
        version_published_at=announced,
        contributors=contributors,
        organizations=organizations,
        countries=_countries(country),
        domains=_domains_from_values(
            [title, str(payload.get("pref_terms") or "")]
        ),
        economic_facts=[fact] if fact.amount is not None else [],
        metadata={
            "project_num": payload.get("project_num"),
            "core_project_num": payload.get("core_project_num"),
            "fiscal_year": fiscal_year,
            "activity_code": payload.get("activity_code"),
            "funding_mechanism": payload.get("funding_mechanism"),
            "agency": agency.get("abbreviation"),
            "project_start": _iso_date(payload.get("project_start_date")),
            "project_end": _iso_date(payload.get("project_end_date")),
            "is_new": payload.get("is_new"),
        },
    )


def parse_nsf(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    """One NSF award as a grant document."""
    award = str(payload["id"])
    document_id = stable_id("document", "nsf", award)
    version_id = stable_id(
        "version", document_id, _observation_hash(payload, "nsf")
    )
    country = payload.get("awardeeCountryCode") or None
    organizations = []
    recipient = None
    name = payload.get("awardeeName") or payload.get("awardee")
    if name:
        recipient = _organization(
            name,
            "nsf",
            str(payload.get("ueiNumber") or name),
            None,
            "grant_recipient",
            country,
            scheme="uei" if payload.get("ueiNumber") else "nsf",
        )
        organizations.append(recipient)
    payer = _organization(
        "National Science Foundation",
        "nsf",
        "agency:NSF",
        "government",
        "funder",
        "US",
    )
    organizations.append(payer)
    announced = _iso_date(payload.get("date") or payload.get("startDate"))
    obligated = _number(payload.get("fundsObligatedAmt"))
    fact = _fact(
        document_id,
        "grant_award",
        obligated,
        "USD",
        announced,
        # Obligated so far; the estimated total is the planned ceiling.
        amount_max=_number(payload.get("estimatedTotalAmt")),
        recipient=recipient.organization_id if recipient else None,
        payer=payer.organization_id,
        source_field="fundsObligatedAmt",
    )
    title = str(payload.get("title") or award)
    chunks = _chunks(
        version_id,
        [("title", title), ("abstract", payload.get("abstractText"))],
    )
    contributors = []
    investigator = " ".join(
        part
        for part in (payload.get("piFirstName"), payload.get("piLastName"))
        if part
    )
    if investigator:
        contributors.append(
            Contributor(
                contributor_id=stable_id(
                    "person", "nsf", str(payload.get("piId") or investigator)
                ),
                name=investigator,
                role="principal_investigator",
            )
        )
    return _document(
        "nsf",
        award,
        payload,
        raw,
        url=f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={award}",
        document_id=document_id,
        version_id=version_id,
        chunks=chunks,
        document_type=DocumentType.GRANT,
        title=title,
        language="en",
        published_at=announced,
        version_published_at=announced,
        contributors=contributors,
        organizations=organizations,
        countries=_countries(country),
        domains=_domains_from_values(
            [title, str(payload.get("fundProgramName") or "")]
        ),
        economic_facts=[fact] if fact.amount is not None else [],
        metadata={
            "program": payload.get("fundProgramName"),
            "directorate": payload.get("orgLongName"),
            "division": payload.get("orgLongName2"),
            "award_type": payload.get("transType"),
            "project_start": _iso_date(payload.get("startDate")),
            "project_end": _iso_date(payload.get("expDate")),
            "estimated_total": _number(payload.get("estimatedTotalAmt")),
        },
    )


# --------------------------------------------------------------- vacancies


def _salary_currency(value: Any) -> Optional[str]:
    """hh.ru writes rubles as RUR, trudvsem as «руб.»."""
    text = str(value or "").strip(" «»").upper()
    if text in ("RUR", "RUB", "РУБ.", "РУБ"):
        return "RUB"
    return text if re.fullmatch(r"[A-Z]{3}", text) else None


def parse_trudvsem(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    """One vacancy of «Работа России» (trudvsem.ru) as a job posting."""
    vacancy = payload.get("vacancy") or payload
    record = str(vacancy["id"])
    document_id = stable_id("document", "trudvsem", record)
    version_id = stable_id(
        "version", document_id, _observation_hash(vacancy, "trudvsem")
    )
    company = vacancy.get("company") or {}
    organizations = []
    employer = None
    if company.get("name") and not company.get("hr-agency"):
        employer = _organization(
            company["name"],
            "trudvsem",
            str(
                company.get("inn")
                or company.get("companycode")
                or company["name"]
            ),
            None,
            "employer",
            "RU",
            scheme="inn" if company.get("inn") else "trudvsem",
        )
        organizations.append(employer)
    posted = _iso_date(vacancy.get("creation-date"))
    fact = _fact(
        document_id,
        "salary_offer",
        _number(vacancy.get("salary_min")),
        _salary_currency(vacancy.get("currency")) or "RUB",
        posted,
        amount_max=_number(vacancy.get("salary_max")),
        period="per_month",
        payer=employer.organization_id if employer else None,
        source_field="salary_min",
    )
    requirement = vacancy.get("requirement") or {}
    skills = [
        str(item.get("name") if isinstance(item, Mapping) else item)
        for item in vacancy.get("skills") or []
    ]
    title = str(vacancy.get("job-name") or record)
    chunks = _chunks(
        version_id,
        [
            ("title", title),
            ("duty", _text(vacancy.get("duty"))),
            ("requirement", _text(requirement.get("qualification"))),
            ("skills", ", ".join(skills)),
        ],
    )
    return _document(
        "trudvsem",
        record,
        payload,
        raw,
        url=vacancy.get("vac_url") or f"https://trudvsem.ru/vacancy/{record}",
        document_id=document_id,
        version_id=version_id,
        chunks=chunks,
        document_type=DocumentType.JOB_POSTING,
        title=title,
        language="ru",
        published_at=posted,
        version_published_at=posted,
        organizations=organizations,
        countries=_countries("RU"),
        domains=_domains_from_values([title, *skills]),
        economic_facts=[fact] if fact.amount is not None else [],
        metadata={
            "region": (vacancy.get("region") or {}).get("name"),
            "specialisation": (vacancy.get("category") or {}).get(
                "specialisation"
            ),
            "experience_years": requirement.get("experience"),
            "education": requirement.get("education"),
            "schedule": vacancy.get("schedule"),
            "skills": skills,
            "work_places": vacancy.get("work_places"),
        },
    )


def parse_hh(
    payload: Mapping[str, Any], raw: Optional[bytes] = None
) -> DocumentEnvelope:
    """One hh.ru vacancy (full card with description) as a job posting."""
    record = str(payload["id"])
    document_id = stable_id("document", "hh", record)
    version_id = stable_id(
        "version", document_id, _observation_hash(payload, "hh")
    )
    employer_data = payload.get("employer") or {}
    organizations = []
    employer = None
    if employer_data.get("name"):
        employer = _organization(
            employer_data["name"],
            "hh",
            str(employer_data.get("id") or employer_data["name"]),
            None,
            "employer",
        )
        organizations.append(employer)
    posted = _iso_date(
        payload.get("published_at") or payload.get("created_at")
    )
    salary = payload.get("salary") or {}
    fact = _fact(
        document_id,
        "salary_offer",
        _number(salary.get("from")) or _number(salary.get("to")),
        _salary_currency(salary.get("currency")),
        posted,
        amount_max=_number(salary.get("to")) if salary.get("from") else None,
        period="per_month",
        payer=employer.organization_id if employer else None,
        source_field="salary",
    )
    skills = [
        str(item.get("name"))
        for item in payload.get("key_skills") or []
        if isinstance(item, Mapping) and item.get("name")
    ]
    snippet = payload.get("snippet") or {}
    title = str(payload.get("name") or record)
    chunks = _chunks(
        version_id,
        [
            ("title", title),
            (
                "description",
                _text(payload.get("description"))
                or "\n".join(
                    _text(snippet.get(key))
                    for key in ("responsibility", "requirement")
                ),
            ),
            ("skills", ", ".join(skills)),
        ],
    )
    return _document(
        "hh",
        record,
        payload,
        raw,
        url=payload.get("alternate_url") or f"https://hh.ru/vacancy/{record}",
        document_id=document_id,
        version_id=version_id,
        chunks=chunks,
        document_type=DocumentType.JOB_POSTING,
        title=title,
        language="ru",
        published_at=posted,
        version_published_at=posted,
        organizations=organizations,
        domains=_domains_from_values([title, *skills]),
        economic_facts=[fact] if fact.amount is not None else [],
        metadata={
            "area": (payload.get("area") or {}).get("name"),
            "experience": (payload.get("experience") or {}).get("name"),
            "salary_gross": salary.get("gross"),
            "skills": skills,
            "professional_roles": [
                item.get("name")
                for item in payload.get("professional_roles") or []
                if isinstance(item, Mapping)
            ],
        },
    )


PARSERS = {
    "nih": parse_nih,
    "nsf": parse_nsf,
    "trudvsem": parse_trudvsem,
    "hh": parse_hh,
}


# --------------------------------------------------------------- discovery


def _page(
    source: str,
    records: List[Mapping[str, Any]],
    identify,
    title_of,
    url_of,
    next_cursor: Optional[str],
    total: Optional[int],
    limitations: Optional[List[dict]] = None,
) -> dict:
    items = []
    for record in records:
        record = _without_private(dict(record))
        source_id = str(identify(record))
        items.append(
            {
                "source": source,
                "source_id": source_id,
                "canonical_id": f"{source}:{source_id}",
                "title": str(title_of(record) or source_id),
                "url": url_of(record),
                "payload": record,
            }
        )
    limitations = limitations or []
    return {
        "items": items,
        "next_cursor": next_cursor if items else None,
        "total": total,
        "complete": (next_cursor is None or not items) and not limitations,
        "limitations": limitations,
    }


def _query(topic: str) -> str:
    return " ".join(str(topic).split())


async def discover_nih(topic: str, cursor: Optional[str] = None) -> dict:
    """NIH RePORTER awards whose title, terms or abstract hold the phrase.

    The API returns at most 15,000 records per query.
    """
    settings = _settings("nih")
    offset = int(cursor) if cursor not in (None, "", "*") else 0
    limit = int(settings["page_size"])
    phrase = _query(topic).replace('"', " ")
    payload = await resolve(
        fetch_json(
            settings["api_base"],
            json_body={
                "criteria": {
                    "advanced_text_search": {
                        "operator": "advanced",
                        "search_field": "projecttitle,terms,abstracttext",
                        "search_text": f'"{phrase}"',
                    }
                },
                "offset": offset,
                "limit": limit,
                "sort_field": "project_start_date",
                "sort_order": "desc",
            },
        )
    )
    records = payload.get("results")
    if not isinstance(records, list):
        raise ValueError("Invalid NIH RePORTER response")
    total = (payload.get("meta") or {}).get("total")
    cap = int(settings["max_records"])
    following = offset + limit
    limitations = []
    if isinstance(total, int) and total > cap and following >= cap:
        limitations.append(
            {
                "code": "nih_search_cap",
                "reported_total": total,
                "accessible_total": cap,
                "message": (
                    f"NIH RePORTER отдаёт не больше {cap} записей на запрос."
                ),
            }
        )
    has_more = (
        len(records) == limit
        and following < cap
        and (not isinstance(total, int) or following < total)
    )
    return _page(
        "nih",
        records,
        lambda record: record["appl_id"],
        lambda record: record.get("project_title"),
        lambda record: record.get("project_detail_url"),
        str(following) if has_more else None,
        total if isinstance(total, int) else None,
        limitations,
    )


async def discover_nsf(topic: str, cursor: Optional[str] = None) -> dict:
    """NSF awards matching the phrase (25 per page, 1-based offset)."""
    settings = _settings("nsf")
    offset = int(cursor) if cursor not in (None, "", "*") else 1
    size = int(settings["page_size"])
    query = urlencode(
        {
            "keyword": f'"{_query(topic)}"',
            "rpp": size,
            "offset": offset,
            "printFields": ",".join(settings["fields"]),
        }
    )
    payload = await resolve(fetch_json(f"{settings['api_base']}?{query}"))
    response = payload.get("response") or {}
    records = response.get("award")
    if records is None and "serviceNotification" in response:
        raise ValueError("NSF Awards API rejected the query")
    records = records or []
    total = (response.get("metadata") or {}).get("totalCount")
    following = offset + size
    has_more = len(records) == size and (
        not isinstance(total, int) or following <= total
    )
    return _page(
        "nsf",
        records,
        lambda record: record["id"],
        lambda record: record.get("title"),
        lambda record: (
            f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={record['id']}"
        ),
        str(following) if has_more else None,
        total if isinstance(total, int) else None,
    )


def _relevant(topic: str, *texts: Any) -> bool:
    """The query occurs in the vacancy as a phrase, up to inflection.

    trudvsem matches single words loosely: "машинное обучение" finds
    "машинист ... машины (с обучением)".
    """
    wanted = lexical_tokens(topic)
    size = len(wanted)
    for text in texts:
        tokens = lexical_tokens(str(text or ""))
        if size and any(
            tokens[start : start + size] == wanted
            for start in range(len(tokens) - size + 1)
        ):
            return True
    return False


async def discover_trudvsem(topic: str, cursor: Optional[str] = None) -> dict:
    """Vacancies of «Работа России» whose text holds every query word."""
    settings = _settings("trudvsem")
    page = int(cursor) if cursor not in (None, "", "*") else 0
    size = int(settings["page_size"])
    query = urlencode({"text": _query(topic), "offset": page, "limit": size})
    payload = await resolve(fetch_json(f"{settings['api_base']}?{query}"))
    results = payload.get("results") or {}
    wrapped = results.get("vacancies") or []
    vacancies = [
        item.get("vacancy") for item in wrapped if isinstance(item, Mapping)
    ]
    vacancies = [item for item in vacancies if isinstance(item, Mapping)]
    total = (payload.get("meta") or {}).get("total")
    relevant = [
        item
        for item in vacancies
        if _relevant(
            topic,
            item.get("job-name"),
            item.get("duty"),
            (item.get("requirement") or {}).get("qualification"),
            " ".join(map(str, item.get("skills") or [])),
        )
    ]
    has_more = len(wrapped) == size and (
        not isinstance(total, int) or (page + 1) * size < total
    )
    page_result = _page(
        "trudvsem",
        relevant,
        lambda record: record["id"],
        lambda record: record.get("job-name"),
        lambda record: record.get("vac_url"),
        str(page + 1) if has_more else None,
        total if isinstance(total, int) else None,
    )
    # An irrelevant page is not the end of the results.
    if not relevant and has_more:
        page_result["next_cursor"] = str(page + 1)
        page_result["complete"] = False
    return page_result


def _hh_headers() -> Dict[str, str]:
    settings = _settings("hh")
    headers = {"HH-User-Agent": settings["user_agent"]}
    token = os.getenv("HH_ACCESS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def discover_hh(topic: str, cursor: Optional[str] = None) -> dict:
    """hh.ru vacancies; the API needs an application token (HH_ACCESS_TOKEN,
    https://dev.hh.ru) and returns at most 2,000 results per query.
    """
    if not os.getenv("HH_ACCESS_TOKEN", "").strip():
        return {
            "items": [],
            "next_cursor": None,
            "total": None,
            "complete": False,
            "limitations": [
                {
                    "code": "hh_token_missing",
                    "message": (
                        "hh.ru требует токен приложения: задайте "
                        "HH_ACCESS_TOKEN (dev.hh.ru)."
                    ),
                }
            ],
        }
    settings = _settings("hh")
    page = int(cursor) if cursor not in (None, "", "*") else 0
    size = int(settings["page_size"])
    query = urlencode({"text": _query(topic), "per_page": size, "page": page})
    payload = await resolve(
        fetch_json(f"{settings['api_base']}?{query}", _hh_headers())
    )
    records = payload.get("items")
    if not isinstance(records, list):
        raise ValueError("Invalid hh.ru response")
    found = payload.get("found")
    cap = int(settings["max_records"])
    has_more = (
        page + 1 < int(payload.get("pages") or 0) and (page + 1) * size < cap
    )
    limitations = []
    if isinstance(found, int) and found > cap and not has_more:
        limitations.append(
            {
                "code": "hh_search_cap",
                "reported_total": found,
                "accessible_total": cap,
                "message": f"hh.ru отдаёт не больше {cap} вакансий на запрос.",
            }
        )
    return _page(
        "hh",
        records,
        lambda record: record["id"],
        lambda record: record.get("name"),
        lambda record: record.get("alternate_url"),
        str(page + 1) if has_more else None,
        min(found, cap) if isinstance(found, int) else None,
        limitations,
    )


async def hydrate_hh(record: Mapping[str, Any]) -> Dict[str, Any]:
    """The full vacancy card (description, key skills) of a search item."""
    settings = _settings("hh")
    payload = await resolve(
        fetch_json(f"{settings['api_base']}/{record['id']}", _hh_headers())
    )
    return _without_private(
        {
            **payload,
            "_retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    )


DISCOVERERS = {
    "nih": discover_nih,
    "nsf": discover_nsf,
    "trudvsem": discover_trudvsem,
    "hh": discover_hh,
}
