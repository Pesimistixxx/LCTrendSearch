"""Point-in-time features of the economic layer, mixed with the
organizations of the rest of the corpus."""

import json

import pytest

from lctrend.graph.temporal import TemporalCorpus
from lctrend.graph.training import build_snapshot_rows


def version(document_id, when, family, organizations, companies=(), facts=()):
    return {
        "document_id": document_id,
        "version_id": document_id + "-v1",
        "document_type": {
            "funding": "grant",
            "labor_market": "job_posting",
        }.get(family, "article"),
        "source_family": family,
        "source_id": family,
        "document_published_at": when,
        "version_published_at": when,
        "retrieved_at": when,
        "coverage": "full_text",
        "organizations": list(organizations),
        "companies": list(companies),
        "economic_facts_json": json.dumps(list(facts)),
        "extracted": True,
        "extracted_at": when,
    }


def grant(when, real, recipient):
    return {
        "category": "grant_award",
        "observed_at": when,
        "amount_usd_real": real,
        "recipient_organization_id": recipient,
        "payer_organization_id": "org:nsf",
    }


def offer(when, low, high, employer):
    return {
        "category": "salary_offer",
        "observed_at": when,
        "amount_usd_real": low,
        "amount_max_usd_real": high,
        "payer_organization_id": employer,
    }


def corpus():
    versions = [
        # Intel publishes on the technology...
        version(
            "paper", "2023-01-01", "scholarly", ["org:intel"], ["org:intel"]
        ),
        # ...is funded for it...
        version(
            "grant-old",
            "2023-03-01",
            "funding",
            ["org:intel", "org:nsf"],
            ["org:intel"],
            [grant("2023-03-01", 100.0, "org:intel")],
        ),
        version(
            "grant-new",
            "2024-06-01",
            "funding",
            ["org:mit", "org:nsf"],
            facts=[grant("2024-06-01", 300.0, "org:mit")],
        ),
        # ...and hires for it; the second offer comes after the snapshot.
        version(
            "vacancy",
            "2024-09-01",
            "labor_market",
            ["org:intel"],
            ["org:intel"],
            [offer("2024-09-01", 2000.0, 3000.0, "org:intel")],
        ),
        version(
            "vacancy-late",
            "2025-06-01",
            "labor_market",
            ["org:sber"],
            facts=[offer("2025-06-01", 9000.0, None, "org:sber")],
        ),
    ]
    return TemporalCorpus(
        {
            "versions": versions,
            "technologies": [{"technology_id": "t", "technology": "RAG"}],
            "mentions": [
                {
                    "technology_id": "t",
                    "version_id": row["version_id"],
                    "observed_at": row["version_published_at"],
                    "mentions": 1,
                    "accepted": 1,
                }
                for row in versions
            ],
        }
    )


def test_money_demand_and_organizations_known_at_the_snapshot():
    (row,) = build_snapshot_rows(corpus(), "2025-01-01")
    assert row["grant_amount_usd_real"] == 400.0
    assert row["grant_amount_last_year_usd_real"] == 300.0
    assert row["grant_median_usd_real"] == 200.0
    assert row["grant_recipient_count"] == 2
    assert row["grant_funder_count"] == 1
    # A quarter of the money goes to a company.
    assert row["grant_company_share"] == pytest.approx(0.25)
    # Intel publishes, is funded and hires: one organization everywhere.
    assert row["funded_producer_count"] == 1
    assert row["hiring_producer_count"] == 1
    # The 2025 offer is not known on 2025-01-01.
    assert row["salary_median_usd_real"] == 2500.0
    assert row["vacancy_count_last_year"] == 1
    assert row["grant_data_missing"] is False
    assert row["grant_amount_usd_real_snapshot_pct"] == 0.5


def test_a_layer_never_searched_nor_found_stays_missing():
    data = corpus()
    (row,) = build_snapshot_rows(data, "2023-02-01")
    assert row["grant_amount_usd_real"] is None
    assert row["salary_median_usd_real"] is None
    assert row["grant_data_missing"] is True
    assert row["vacancy_data_missing"] is True
