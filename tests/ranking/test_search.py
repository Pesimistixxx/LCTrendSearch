"""/api/search response built from the graph, in the frontend contract."""

import copy
from datetime import date

import pytest

from lctrend.core.config import load_catalog
from lctrend.core.models import stable_id
from lctrend.graph.temporal import TemporalCorpus
from lctrend.ranking.search import (
    _search_words,
    match_domains,
    search_response,
)

T = date(2021, 1, 1)
FINTECH = stable_id("domain", "Fintech")
ROBOTICS = stable_id("domain", "Robotics")

# Keys the React screens read (frontend/src/api.js contract).
SIGNAL_KEYS = {
    "id",
    "title",
    "domain",
    "score",
    "relevanceScore",
    "weakSignalScore",
    "semanticSimilarity",
    "bm25Score",
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
    assert "Machine learning" in match_domains("технологии в ИИ")
    assert "Cybersecurity" in match_domains(
        "слабые сигналы в кибербезопасности"
    )
    assert match_domains("что-то непонятное") == {}


def test_search_tokens_drop_generic_request_words_but_keep_cplusplus():
    assert _search_words("Слабые сигналы для C++") == ["c++"]


def test_search_returns_the_frontend_contract_for_a_domain():
    response = search_response(corpus(), "решения в финтехе", T, settings())
    assert response["demo"] is False
    assert response["snapshot"] == "2021-01-01"
    assert response["scope"] == "lexical"
    assert [signal["id"] for signal in response["signals"]] == ["pay", "kyc"]
    assert response["stats"] == {
        "sourcesProcessed": 10,
        "candidates": 2,
        "confident": sum(
            signal["weakSignalScore"] > 0.75
            for signal in response["signals"]
        ),
    }
    signal = response["signals"][0]
    assert set(signal) >= SIGNAL_KEYS
    assert 0 < signal["score"] < 1
    assert signal["relevanceScore"] > 0
    assert signal["weakSignalScore"] > 0
    assert signal["semanticSimilarity"] is None
    assert signal["bm25Score"] > 0
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


def test_unmatched_query_does_not_show_unrelated_top_candidates():
    response = search_response(corpus(), "что-то непонятное", T, settings())
    assert response["scope"] == "lexical"
    assert response["note"]
    assert response["signals"] == []


def test_generic_query_requires_an_area_or_technology():
    response = search_response(corpus(), "слабые сигналы", T, settings())
    assert response["signals"] == []
    assert "Уточните" in response["note"]


def test_label_words_select_technologies_without_a_domain_match():
    response = search_response(corpus(), "soft grippers", T, settings())
    assert response["scope"] == "lexical"
    assert [signal["id"] for signal in response["signals"]] == ["arm"]


def test_model_probability_is_the_signal_score_and_noise_is_rejected():
    labels = {
        "pay": {
            "probability": 0.0,
            "is_technology": True,
        },
        "kyc": {
            "probability": 0.95,
            "flag": True,
            "model": "stacked.cbm",
            "snapshot": "2020-10-01",
            "verdict": "success",
            "is_technology": True,
            "llm_score": 0.2,
            "rationale": "Rising from niche papers to pilots.",
        },
    }
    response = search_response(
        corpus(), "fintech", T, settings(), labels=labels
    )
    assert response["ranking"] == "model"
    # Same relevance to «fintech»: the model's probability decides.
    assert [signal["id"] for signal in response["signals"]] == ["kyc", "pay"]
    kyc, pay = response["signals"]
    assert kyc["weakSignalScore"] == 0.95 and pay["weakSignalScore"] == 0.0
    assert kyc["model"]["verdict"] == "состоялась"
    assert kyc["model"]["flag"] is True
    assert "Rising from niche papers" in kyc["whyWeak"]
    assert "вероятность обученной модели" in kyc["confidenceReason"]
    assert response["stats"]["confident"] == 1


def test_noise_matching_the_query_is_rejected_not_shown():
    labels = {
        "kyc": {
            "verdict": "junk",
            "is_technology": False,
            "rationale": "A product feature, not a technology.",
        },
    }
    response = search_response(
        corpus(), "fintech", T, settings(), labels=labels
    )
    assert [signal["id"] for signal in response["signals"]] == ["pay"]
    assert response["ranking"] == "rule"
    assert response["rejected"][0] == {
        "title": "Zero-knowledge KYC",
        "category": "noise",
        "reason": "LLM: не технология. A product feature, not a technology.",
    }


def test_without_labels_the_rule_score_is_kept():
    response = search_response(corpus(), "решения в финтехе", T, settings())
    assert response["ranking"] == "rule"
    assert all(signal["model"] is None for signal in response["signals"])


def test_search_service_falls_back_when_labels_fail():
    import asyncio

    from lctrend.ranking.search import SearchService

    async def read():
        return {"versions": [], "mentions": [], "technologies": []}

    async def broken():
        raise ConnectionError("graph down")

    service = SearchService(read, settings(), read_labels=broken)
    response = asyncio.run(service.search("anything", T))
    assert response["ranking"] == "rule"


def test_exact_mature_technology_shows_its_rejection_reason():
    response = search_response(
        corpus(), "double-entry ledger", T, settings()
    )
    assert response["signals"] == []
    assert response["rejected"][0]["category"] == "mature"


def test_semantic_query_finds_technology_without_shared_words():
    sample = corpus()
    sample.embeddings = {"pay": [1.0, 0.0], "kyc": [0.0, 1.0]}
    sample.embedding_model = "EmbeddingsGigaR"
    response = search_response(
        sample, "orbital humming", T, settings(),
        query_embedding=[1.0, 0.0],
    )
    assert response["scope"] == "hybrid"
    assert [signal["id"] for signal in response["signals"]] == ["pay"]
    assert response["signals"][0]["semanticSimilarity"] == 1.0


def test_relevance_and_signal_score_are_separate_and_top_k_applies():
    sample = corpus()
    sample.embeddings = {"pay": [1.0, 0.0], "kyc": [0.8, 0.6]}
    sample.embedding_model = "EmbeddingsGigaR"
    config = settings()
    config["top_k"] = 1
    response = search_response(
        sample, "fintech", T, config, query_embedding=[1.0, 0.0]
    )
    assert len(response["signals"]) == 1
    first = response["signals"][0]
    assert first["id"] == "pay"
    assert first["score"] == round(
        0.75 * first["relevanceScore"] + 0.25 * first["weakSignalScore"],
        4,
    )


def test_search_returns_at_most_fifteen_of_many_matching_technologies():
    data = {"versions": [], "mentions": [], "technologies": []}
    for index in range(20):
        technology(
            data, f"pay-{index}", f"Programmable payments {index}",
            FINTECH, [("2020-01-01", "a"), ("2020-10-01", "b")],
        )
    response = search_response(
        TemporalCorpus(data), "fintech", T, settings()
    )
    assert response["stats"]["candidates"] == 20
    assert len(response["signals"]) == 15


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


def test_search_service_embeds_query_in_the_graphs_gigachat_space():
    import asyncio

    from lctrend.ranking.search import SearchService

    data = {"versions": [], "mentions": [], "technologies": []}
    docs = [("2020-01-01", "a"), ("2020-10-01", "b")]
    technology(data, "pay", "Programmable payments", FINTECH, docs)
    technology(data, "arm", "Soft grippers", ROBOTICS, docs)
    for row in data["technologies"]:
        row["embedding_model"] = "EmbeddingsGigaR"
        row["embedding"] = (
            [1.0, 0.0] if row["technology_id"] == "pay" else [0.0, 1.0]
        )
    calls = []

    async def embed(query, model):
        calls.append((query, model))
        return [1.0, 0.0]

    async def read():
        return data

    result = asyncio.run(
        SearchService(read, settings(), embed_query=embed).search(
            "orbital humming", T
        )
    )
    assert calls == [("orbital humming", "EmbeddingsGigaR")]
    assert [signal["id"] for signal in result["signals"]] == ["pay"]


def test_search_rejects_vectors_from_a_different_embedding_model():
    import asyncio

    from lctrend.ranking.search import EmbeddingIndexError, SearchService

    data = {"versions": [], "mentions": [], "technologies": []}
    technology(
        data, "pay", "Programmable payments", FINTECH,
        [("2020-01-01", "a"), ("2020-10-01", "b")],
    )
    data["technologies"][0].update(
        embedding=[1.0, 0.0], embedding_model="other-model"
    )

    async def read():
        return data

    async def embed(query, model):
        pytest.fail("An incompatible model must not be queried")

    with pytest.raises(EmbeddingIndexError):
        asyncio.run(
            SearchService(read, settings(), embed_query=embed).search(
                "payments", T
            )
        )


def test_search_falls_back_to_bm25_when_gigachat_is_unavailable():
    import asyncio

    from lctrend.llm.client import LLMError
    from lctrend.ranking.search import SearchService

    data = {"versions": [], "mentions": [], "technologies": []}
    technology(
        data, "pay", "Programmable payments", FINTECH,
        [("2020-01-01", "a"), ("2020-10-01", "b")],
    )
    data["technologies"][0].update(
        embedding=[1.0, 0.0], embedding_model="EmbeddingsGigaR"
    )

    async def read():
        return data

    async def embed(query, model):
        raise LLMError("configuration", "no key")

    result = asyncio.run(
        SearchService(read, settings(), embed_query=embed).search(
            "payments", T
        )
    )
    assert result["scope"] == "lexical"
    assert "BM25" in result["note"]
    assert [signal["id"] for signal in result["signals"]] == ["pay"]
