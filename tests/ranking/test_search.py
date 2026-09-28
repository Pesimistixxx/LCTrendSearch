"""/api/search response built from the graph, in the frontend contract."""

import copy
from datetime import date

from lctrend.core.config import load_catalog
from lctrend.core.models import stable_id
from lctrend.graph.temporal import TemporalCorpus
from lctrend.ranking.search import match_domains, search_response

T = date(2021, 1, 1)
FINTECH = stable_id("domain", "Fintech")
ROBOTICS = stable_id("domain", "Robotics")

# Keys the React screens read (frontend/src/api.js contract).
SIGNAL_KEYS = {
    "id",
    "title",
    "domain",
    "score",
    "stage",
    "summary",
    "predictors",
    "trend",
    "description",
    "advantages",
    "cases",
    "reports",
    "quotes",
    "whyWeak",
    "confidenceReason",
    "sources",
}


def technology(data, technology_id, label, domain, documents):
    data["technologies"].append(
        {"technology_id": technology_id, "technology": label}
    )
    for index, (published, group) in enumerate(documents):
        document = f"{technology_id}-{index}"
        data["versions"].append(
            {
                "document_id": document,
                "version_id": document + "-v1",
                "document_type": "article",
                "source_family": "scholarly",
                "source_id": "source:openalex",
                "reliability_tier": 3,
                "document_published_at": published,
                "version_published_at": published,
                "independence_group": group,
                "domains": [domain],
                "title": f"Paper {document}",
                "url": f"https://example.org/{document}",
            }
        )
        data["mentions"].append(
            {
                "technology_id": technology_id,
                "version_id": document + "-v1",
                "observed_at": published,
                "mentions": 2,
            }
        )


def corpus():
    data = {"versions": [], "mentions": [], "technologies": []}
    recent = [("2019-06-01", "a"), ("2020-03-01", "b"), ("2020-09-01", "c")]
    technology(data, "pay", "Programmable payments", FINTECH, recent)
    technology(
        data,
        "kyc",
        "Zero-knowledge KYC",
        FINTECH,
        [("2020-01-01", "d"), ("2020-10-01", "e")],
    )
    technology(data, "arm", "Soft grippers", ROBOTICS, recent)
    technology(
        data,
        "ledger",
        "Double-entry ledger",
        FINTECH,
        [("2001-01-01", "f"), ("2020-02-01", "g")],
    )
    data["assertions"] = [
        {
            "technology_id": "pay",
            "assertion_id": "claim",
            "version_id": "pay-2-v1",
            "observed_at": "2020-09-01",
            "status": "accepted",
            "quote": "Programmable payments cut settlement to seconds.",
        }
    ]
    return TemporalCorpus(data)


def settings():
    value = copy.deepcopy(load_catalog("ranking"))
    value["include_novelty"] = False
    return value


def test_query_words_match_domain_aliases_and_child_domains():
    assert match_domains("перспективные решения в финтехе") == {
        "Fintech": FINTECH,
    }
    assert "Machine learning" in match_domains(
        "artificial intelligence trends"
    )
    assert "Cybersecurity" in match_domains(
        "слабые сигналы в кибербезопасности"
    )
    assert match_domains("что-то непонятное") == {}


def test_search_returns_the_frontend_contract_for_a_domain():
    response = search_response(corpus(), "решения в финтехе", T, settings())
    assert response["demo"] is False
    assert response["snapshot"] == "2021-01-01"
    assert response["scope"] == "domain"
    assert [signal["id"] for signal in response["signals"]] == ["pay", "kyc"]
    assert response["stats"] == {
        "sourcesProcessed": 10,
        "candidates": 2,
        "confident": sum(
            signal["score"] > 0.75 for signal in response["signals"]
        ),
    }
    signal = response["signals"][0]
    assert set(signal) >= SIGNAL_KEYS
    assert 0 < signal["score"] < 1
    assert signal["domain"] == "Fintech"
    assert signal["stage"] == "не определена"
    assert len(signal["predictors"]) == 3
    assert {"name", "weight", "value"} <= set(signal["predictors"][0])
    # Mentions per quarter up to T: 2019-06, 2020-03 and 2020-09.
    assert len(signal["trend"]) == 10 and sum(signal["trend"]) == 6
    assert signal["trend"][-2:] == [2, 0]
    assert signal["quotes"] == [
        {
            "text": "Programmable payments cut settlement to seconds.",
            "title": "Paper pay-2",
            "date": "2020-09-01",
            "url": "https://example.org/pay-2",
        }
    ]
    assert signal["sources"][0] == {
        "title": "Paper pay-2",
        "url": "https://example.org/pay-2",
        "date": "2020-09-01",
        "type": "Научная публикация",
        "lang": "—",
        "trust": "high",
    }
    assert signal["advantages"] == signal["cases"] == signal["reports"] == []
    rejected = response["rejected"]
    assert rejected == [
        {
            "title": "Double-entry ledger",
            "category": "mature",
            "reason": rejected[0]["reason"],
        }
    ]
    assert len(response["signals"]) <= settings()["top_k"]


def test_unmatched_query_falls_back_to_all_candidates_and_says_so():
    response = search_response(corpus(), "что-то непонятное", T, settings())
    assert response["scope"] == "all"
    assert response["note"]
    assert {signal["id"] for signal in response["signals"]} == {
        "pay",
        "kyc",
        "arm",
    }


def test_label_words_select_technologies_without_a_domain_match():
    response = search_response(corpus(), "soft grippers", T, settings())
    assert response["scope"] == "label"
    assert [signal["id"] for signal in response["signals"]] == ["arm"]


def test_search_service_reads_the_graph_once_per_cache_period():
    import asyncio

    from lctrend.ranking.search import SearchService

    reads, now = [], [0.0]

    async def read():
        reads.append(1)
        return {
            "versions": corpus_versions(),
            "mentions": corpus_mentions(),
            "technologies": [],
        }

    def corpus_versions():
        return [
            {
                "document_id": "d",
                "version_id": "d-v1",
                "document_type": "article",
                "version_published_at": "2020-06-01",
            }
        ]

    def corpus_mentions():
        return [
            {
                "technology_id": "t",
                "version_id": "d-v1",
                "observed_at": "2020-06-01",
                "mentions": 1,
            }
        ]

    service = SearchService(read, settings(), clock=lambda: now[0])

    async def run():
        first = await service.search("anything", T)
        await service.search("other", T)
        now[0] = 10_000.0
        await service.search("again", T)
        return first

    first = asyncio.run(run())
    assert first["snapshot"] == "2021-01-01"
    assert first["stats"]["sourcesProcessed"] == 1
    assert len(reads) == 2
