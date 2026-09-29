"""Organizations, countries, taxonomy and maturity from source text."""

import asyncio
import re

import pytest

from lctrend.core.models import (
    Artifact,
    Chunk,
    ConceptKind,
    DocumentEnvelope,
    DocumentType,
    SourceRef,
)
from lctrend.graph.store import GraphStore, evidence_chunk_ids
from lctrend.llm.client import ReplayProvider
from lctrend.llm.context import PipelineSettings
from lctrend.llm.contracts import Extraction
from lctrend.llm.pipeline import process_document
from lctrend.llm.validation import validate_local_extraction

pytestmark = pytest.mark.legacy_technology_entities

TEXT = (
    "Acme Energy GmbH developed the solid-state battery in Германии. "
    "The solid-state battery reached TRL 6 in a pilot plant."
)


def document():
    return DocumentEnvelope(
        document_id="doc",
        document_version_id="version",
        document_type=DocumentType.REPORT,
        title="Battery report",
        published_at="2026-03-01",
        source=SourceRef(
            source_id="fixture",
            name="fixture",
            source_type="test",
            record_id="1",
        ),
        artifact=Artifact(
            uri="memory://fixture", sha256="a" * 64, media_type="text/plain"
        ),
        chunks=[Chunk(chunk_id="c1", kind="paragraph", text=TEXT, order=0)],
    )


def entity(local_id, label, kind, quote=None, **extra):
    return {
        "local_id": local_id,
        "label": label,
        "kind": kind,
        "evidence": [{"chunk_id": "c1", "quote": quote or label}],
        **extra,
    }


def claim(claim_id, predicate, roles, quote, **extra):
    return {
        "claim_id": claim_id,
        "predicate": predicate,
        "roles": roles,
        "polarity": "affirmed",
        "modality": "reported",
        "evidence": [{"chunk_id": "c1", "quote": quote}],
        **extra,
    }


FIRST = "Acme Energy GmbH developed the solid-state battery in Германии."
SECOND = "The solid-state battery reached TRL 6 in a pilot plant."


def extraction(**overrides):
    value = {
        "entities": [
            entity(
                "battery",
                "solid-state battery",
                "Technology",
                "the solid-state battery in",
            ),
            entity("acme", "Acme Energy GmbH", "Company"),
            entity("de", "Германии", "Country", country_code="DE"),
        ],
        "claims": [
            claim(
                "developer",
                "developed_by",
                {"subject": "battery", "organization": "acme"},
                FIRST,
            ),
            claim(
                "country",
                "developed_in",
                {"subject": "battery", "country": "de"},
                FIRST,
            ),
            claim(
                "stage",
                "reports_maturity_stage",
                {"subject": "battery"},
                SECOND,
                qualifiers={"stage": "pilot", "trl": 6},
            ),
        ],
        "context_requests": [],
    }
    value.update(overrides)
    return value


def supported(*claim_ids):
    return {
        "items": [
            {"claim_id": item, "decision": "supported", "reason": "Stated."}
            for item in claim_ids
        ]
    }


def settings():
    return PipelineSettings(
        primary_chunks=1,
        max_context_chunks=2,
        max_retries=0,
        retry_delay_seconds=0,
        max_retry_delay_seconds=0,
    )


def issues_for(response, key):
    _, issues = validate_local_extraction(
        document(), Extraction.model_validate(response), {"c1"}
    )
    return issues.get(key, [])


def test_trl_must_be_written_in_the_quote_and_stage_must_be_known():
    response = extraction()
    response["claims"][2]["qualifiers"] = {"stage": "pilot", "trl": 7}
    assert "qualifier_not_grounded:trl" in issues_for(response, "claim:stage")
    response["claims"][2]["qualifiers"] = {"stage": "scaled", "trl": 6}
    assert "invalid_qualifier:stage" in issues_for(response, "claim:stage")
    response["claims"][2]["qualifiers"] = {"trl": 6}
    assert "missing_required_qualifier:stage" in issues_for(
        response, "claim:stage"
    )
    response["claims"][2]["qualifiers"] = {"stage": "pilot"}
    assert issues_for(response, "claim:stage") == []


def test_market_event_needs_a_known_event_and_round():
    response = extraction()
    response["claims"].append(
        claim(
            "event",
            "reports_market_event",
            {"subject": "battery", "organization": "acme"},
            FIRST,
            qualifiers={"event": "product_launch"},
        )
    )
    assert issues_for(response, "claim:event") == []
    response["claims"][3]["qualifiers"] = {
        "event": "funding_round",
        "round": "series_a",
    }
    assert issues_for(response, "claim:event") == []
    response["claims"][3]["qualifiers"] = {"event": "rumor"}
    assert "invalid_qualifier:event" in issues_for(response, "claim:event")
    response["claims"][3]["qualifiers"] = {
        "event": "funding_round",
        "round": "series_z",
    }
    assert "invalid_qualifier:round" in issues_for(response, "claim:event")
    response["claims"][3]["qualifiers"] = {}
    assert "missing_required_qualifier:event" in issues_for(
        response, "claim:event"
    )


def test_country_code_is_required_shape_and_only_for_countries():
    response = extraction()
    response["entities"][2]["country_code"] = "Germany"
    assert "invalid_country_code" in issues_for(response, "entity:de")
    response = extraction()
    response["entities"][1]["country_code"] = "DE"
    assert "country_code_on_non_country" in issues_for(response, "entity:acme")


def test_reviewed_party_claims_become_evidence_backed_graph_edges():
    doc = document()
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider(
                [extraction(), supported("developer", "country", "stage")]
            ),
            settings=settings(),
        )
    )
    assert result.run.status == "succeeded"
    country = next(
        concept
        for concept in result.concepts
        if concept.kind == ConceptKind.COUNTRY
    )
    # The ISO code identifies the country across languages.
    assert country.preferred_label == "DE"

    links = GraphStore._projection_links(doc, result)
    assert {link["relationship"] for link in links} == {
        "DEVELOPED_BY",
        "DEVELOPED_IN",
    }
    developer = next(
        link for link in links if link["relationship"] == "DEVELOPED_BY"
    )
    assert (developer["source_label"], developer["target_label"]) == (
        "Technology",
        "Company",
    )
    assert developer["quote"] == FIRST
    maturity = GraphStore._maturity_evidence(result)
    assert [
        (row["stage"], row["stage_rank"], row["trl"]) for row in maturity
    ] == [("pilot", 4, 6)]


def test_unreviewed_or_planned_party_claims_are_not_projected():
    doc = document()
    response = extraction()
    response["claims"][0]["modality"] = "planned"
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider(
                [response, supported("developer", "country", "stage")]
            ),
            settings=settings(),
        )
    )
    relationships = {
        link["relationship"]
        for link in GraphStore._projection_links(doc, result)
    }
    assert "DEVELOPED_BY" not in relationships
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider([extraction(), supported("country", "stage")]),
            settings=settings(),
        )
    )
    assert not GraphStore._projection_links(doc, result)
    assert not GraphStore._maturity_evidence(result)


class Result:
    def consume(self):
        return None

    def __iter__(self):
        # Stored-kind probe of _settle_family_kinds: no stored nodes.
        return iter(())


class Transaction:
    def __init__(self):
        self.queries = []

    def run(self, query, **parameters):
        assert set(re.findall(r"\$(\w+)", query)).issubset(parameters)
        self.queries.append((query, parameters))
        return Result()


def test_graph_write_uses_labels_batches_and_links_text_countries():
    doc = document()
    result = asyncio.run(
        process_document(
            doc,
            ReplayProvider(
                [extraction(), supported("developer", "country", "stage")]
            ),
            settings=settings(),
        )
    )
    tx = Transaction()
    # As write_processed does: the chunks the extraction stands on.
    asyncio.run(
        GraphStore._write_document(
            tx, doc, kept_chunk_ids=evidence_chunk_ids(result)
        )
    )
    asyncio.run(GraphStore._write_extraction(tx, doc, result))
    queries = [query for query, _ in tx.queries]
    # Every concept match is labeled, so it can use the concept_id constraint.
    assert not any("MATCH (concept {" in query for query in queries)
    assert not any("MATCH (c {concept_id" in query for query in queries)
    assert any(
        "UNWIND $rows AS row" in query and "MENTIONS" in query
        for query in queries
    )
    assert any(
        "UNWIND $rows AS row" in query and "MERGE (c:Chunk" in query
        for query in queries
    )
    same_as = next(
        parameters for query, parameters in tx.queries if "SAME_AS" in query
    )
    assert [row["code"] for row in same_as["rows"]] == ["DE"]
    assert any("DEVELOPED_BY" in query for query in queries)
    assert any("HAS_MATURITY_EVIDENCE" in query for query in queries)
