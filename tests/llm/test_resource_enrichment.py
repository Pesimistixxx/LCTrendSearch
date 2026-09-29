"""Evidence gates for domain, geographic and financial enrichment."""

import pytest

from lctrend.core.config import load_catalog
from lctrend.core.models import (
    Artifact,
    Chunk,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.llm.contracts import (
    Extraction,
    LocalClaim,
    LocalEntity,
    SourceSpan,
)
from lctrend.llm.validation import validate_local_extraction

pytestmark = pytest.mark.legacy_technology_entities


def document(text):
    return DocumentEnvelope(
        document_id="d",
        document_version_id="v",
        document_type=DocumentType.ARTICLE,
        title="Evidence fixture",
        source=SourceRef(
            source_id="s", name="fixture", source_type="test", record_id="1"
        ),
        artifact=Artifact(
            uri="memory://enrichment", sha256="a" * 64,
            media_type="text/plain",
        ),
        chunks=[Chunk(chunk_id="c", kind="paragraph", text=text, order=0)],
    )


def entity(local_id, label, kind, **fields):
    return LocalEntity(
        local_id=local_id,
        label=label,
        kind=kind,
        evidence=[SourceSpan(chunk_id="c", quote=label)],
        **fields,
    )


def extraction(doc, predicate, target=None, **claim_fields):
    entities = [entity("sensor", "Sensor S", ConceptKind.TECHNOLOGY)]
    roles = {"subject": "sensor"}
    if target:
        role, item = target
        entities.append(item)
        roles[role] = item.local_id
    return Extraction(
        entities=entities,
        claims=[LocalClaim(
            claim_id="a", predicate=predicate, roles=roles,
            polarity="affirmed", modality="reported",
            evidence=[SourceSpan(chunk_id="c", quote=doc.chunks[0].text)],
            **claim_fields,
        )],
    )


@pytest.mark.parametrize(
    "predicate,role,kind,label,text",
    [
        ("belongs_to_domain", "domain", ConceptKind.DOMAIN, "robotics",
         "Sensor S is a technology in robotics."),
        ("targets_market", "market", ConceptKind.MARKET_SEGMENT,
         "industrial sensor market",
         "Sensor S targets the industrial sensor market."),
        ("manufactured_in", "country", ConceptKind.COUNTRY, "Germany",
         "Sensor S is manufactured in Germany."),
        ("tested_in", "country", ConceptKind.COUNTRY, "Germany",
         "Sensor S was tested in Germany."),
        ("deployed_in", "country", ConceptKind.COUNTRY, "Germany",
         "Sensor S was deployed in Germany."),
    ],
)
def test_new_roles_accept_anchored_existing_kinds(
    predicate, role, kind, label, text,
):
    doc = document(text)
    fields = {"country_code": "DE"} if kind == ConceptKind.COUNTRY else {}
    candidate = extraction(
        doc, predicate, (role, entity("target", label, kind, **fields))
    )
    anchored, issues = validate_local_extraction(doc, candidate, ["c"])
    assert issues == {}
    assert anchored.claims[0].evidence[0].start == 0
    assert candidate.claims[0].evidence[0].start is None


def test_application_context_cannot_silently_become_domain():
    doc = document("Sensor S is used for monitoring.")
    candidate = extraction(
        doc, "belongs_to_domain",
        ("domain", entity("target", "monitoring",
                          ConceptKind.APPLICATION_CONTEXT)),
    )
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert "role_type_mismatch:domain" in issues["claim:a"]


@pytest.mark.parametrize("code", [None, "", "ZZ", "EU", "XK", "de"])
def test_country_requires_assigned_iso_code(code):
    doc = document("Sensor S was tested in Germany.")
    candidate = extraction(
        doc, "tested_in",
        ("country", entity("target", "Germany", ConceptKind.COUNTRY,
                           country_code=code)),
    )
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert "entity:target" in issues
    assert "invalid_entity:target" in issues["claim:a"]


def test_country_catalog_is_complete_unique_and_not_placeholder_codes():
    codes = load_catalog("countries")["iso_alpha2"]
    assert len(codes) == len(set(codes)) == 249
    assert {"DE", "RU", "CN", "US", "GB", "AX", "SS"} <= set(codes)
    assert not {"ZZ", "EU", "XK"} & set(codes)


@pytest.mark.parametrize("text,trl,valid", [
    ("Sensor S was tested on 6 devices.", 6, False),
    ("Sensor S has TRL 5 and was tested on 6 devices.", 6, False),
    ("Sensor S has TRL 6.", 6, True),
    ("Sensor S has Technology Readiness Level: 6.", 6, True),
    ("Sensor S достиг УГТ 6.", 6, True),
    ("Sensor S достиг уровня технологической готовности 6.", 6, True),
    ("Sensor S has TRL 6.5.", 6, False),
    ("Sensor S has TRL 16.", 6, False),
    ("Sensor S has TRL 10.", 10, False),
])
def test_trl_number_must_belong_to_readiness_marker(text, trl, valid):
    doc = document(text)
    candidate = extraction(
        doc, "reports_maturity_stage",
        qualifiers={"stage": "prototype", "trl": trl},
    )
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert (not issues) == valid


FINANCIAL_PREDICATES = [
    "reported_cost", "reported_investment", "reported_revenue",
    "reported_market_size", "reported_savings",
]


def financial_candidate(value=None, predicate="reported_cost"):
    doc = document(
        "Sensor S serves the industrial sensor market. "
        "The source reports USD 5 million per device in 2025."
    )
    target = (
        "market", entity("market", "industrial sensor market",
                         ConceptKind.MARKET_SEGMENT)
    ) if predicate == "reported_market_size" else None
    return doc, extraction(
        doc, predicate, target,
        values=[value if value is not None else {
            "raw": "USD 5 million", "value": 5, "currency": "USD",
            "unit": "per device", "period": "2025",
        }],
    )


@pytest.mark.parametrize("predicate", FINANCIAL_PREDICATES)
def test_financial_contract_preserves_source_number_and_conditions(predicate):
    doc, candidate = financial_candidate(predicate=predicate)
    anchored, issues = validate_local_extraction(doc, candidate, ["c"])
    assert issues == {}
    assert anchored.claims[0].values == candidate.claims[0].values


@pytest.mark.parametrize("predicate", FINANCIAL_PREDICATES)
def test_financial_assertions_require_measurements(predicate):
    doc, candidate = financial_candidate(predicate=predicate)
    candidate.claims[0].values = []
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert "measurement_values_missing" in issues["claim:a"]


@pytest.mark.parametrize("field", ["raw", "value", "currency"])
def test_missing_financial_fields_are_rejected(field):
    doc, candidate = financial_candidate()
    del candidate.claims[0].values[0][field]
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert "missing_value_field:" + field in issues["claim:a"]


@pytest.mark.parametrize("field,value,expected", [
    ("currency", "EUR", "currency_not_grounded"),
    ("currency", "FAKE", "invalid_currency"),
    ("period", "2028", "value_field_not_grounded:period"),
    ("unit", "per kg", "value_field_not_grounded:unit"),
    ("value", 5000000, "number_not_in_raw_value"),
    ("value", "high", "invalid_numeric_value"),
    ("value", float("inf"), "nonfinite_numeric_value"),
])
def test_financial_units_dates_currency_and_derived_numbers_are_not_invented(
    field, value, expected,
):
    doc, candidate = financial_candidate()
    candidate.claims[0].values[0][field] = value
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert expected in issues["claim:a"]


def test_bare_dollar_does_not_prove_usd_and_no_date_is_guessed():
    doc = document("Sensor S costs $5.")
    candidate = extraction(
        doc, "reported_cost",
        values=[{"raw": "$5", "value": 5, "currency": "$"}],
    )
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert issues == {}
    candidate.claims[0].values[0]["currency"] = "USD"
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert "currency_not_grounded" in issues["claim:a"]


def test_financial_predicate_cannot_use_company_as_technology_subject():
    doc, candidate = financial_candidate()
    candidate.entities[0].kind = ConceptKind.COMPANY
    _, issues = validate_local_extraction(doc, candidate, ["c"])
    assert "role_type_mismatch:subject" in issues["claim:a"]
