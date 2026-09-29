"""AI-proposed crawl topics: cleaning, exclusion and queueing."""

import asyncio

from fastapi.testclient import TestClient

from frontend.server.app import create_app
from lctrend.ingest.topics import (
    SuggestedTopic,
    TopicSuggestions,
    clean_topics,
    suggest_topics,
)
from lctrend.llm.client import LLMError
from tests.server.test_ingest_web import Crawls, Manager


class Provider:
    def __init__(self, topics=None, error=None):
        self.topics, self.error, self.calls = topics or [], error, []

    async def generate(self, schema, system, payload, *, stage="extract"):
        self.calls.append((schema, payload, stage))
        if self.error:
            raise self.error
        return TopicSuggestions(
            topics=[SuggestedTopic(query=q, why="сигнал") for q in self.topics]
        )


def test_topics_are_trimmed_deduplicated_and_capped():
    topics = [
        SuggestedTopic(query=text)
        for text in [
            '  "Sodium-ion  battery" ',
            "sodium ion battery",
            "retrieval-augmented generation",
            "x" * 200,
            "perovskite solar cell",
            "digital twin",
        ]
    ]
    cleaned = clean_topics(topics, 3, exclude=["Perovskite solar cell"])
    assert [item["query"] for item in cleaned] == [
        "Sodium-ion battery",
        "retrieval-augmented generation",
        "digital twin",
    ]


def test_model_gets_direction_count_and_known_topics():
    provider = Provider(["solid-state battery", "known topic"])
    result = asyncio.run(
        suggest_topics(provider, " storage ", 50, exclude=["Known topic", ""])
    )
    assert [item["query"] for item in result] == ["solid-state battery"]
    schema, payload, stage = provider.calls[0]
    assert schema is TopicSuggestions and stage == "review"
    assert payload == {
        "direction": "storage",
        "count": 30,
        "exclude": ["Known topic"],
    }


def app_with(provider, crawls):
    manager = Manager()
    manager._provider_factory = lambda: provider
    return TestClient(
        create_app(
            manager,
            crawl_manager=crawls,
            status_reader=lambda: {"ok": True},
        ),
        base_url="http://localhost",
    )


def test_endpoint_proposes_without_queueing_by_default():
    crawls = Crawls()
    crawls.create(topic="digital twin")
    provider = Provider(["edge ai", "digital twin"])
    with app_with(provider, crawls) as client:
        response = client.post(
            "/api/ingest/topics/suggest",
            json={"direction": "промышленный ИИ", "count": 5},
        )
    assert response.status_code == 200
    assert response.json()["topics"] == [{"query": "edge ai", "why": "сигнал"}]
    assert response.json()["crawl_ids"] == []
    assert provider.calls[0][1]["exclude"] == ["digital twin"]
    assert len(crawls.crawls) == 1


def test_endpoint_queues_one_crawl_per_topic():
    crawls = Crawls()
    provider = Provider(["edge ai", "tinyml"])
    with app_with(provider, crawls) as client:
        response = client.post(
            "/api/ingest/topics/suggest",
            json={
                "direction": "ИИ на устройствах",
                "queue": True,
                "limit": 20,
            },
        )
    assert response.status_code == 200
    assert [(c["topic"], c["limit"]) for c in crawls.crawls] == [
        ("edge ai", 20),
        ("tinyml", 20),
    ]
    assert len(response.json()["crawl_ids"]) == 2


def test_model_failure_is_reported_without_details():
    provider = Provider(error=LLMError("http_error", "secret body"))
    with app_with(provider, Crawls()) as client:
        response = client.post(
            "/api/ingest/topics/suggest", json={"direction": "storage"}
        )
        blank = client.post(
            "/api/ingest/topics/suggest", json={"direction": "   "}
        )
    assert response.status_code == 502
    assert "secret" not in response.text
    assert blank.status_code == 422


def test_presets_list_the_hundred_test_signals():
    from lctrend.core.config import load_catalog

    shipped = load_catalog("topic_presets")
    queries = [
        item["query"]
        for group in shipped["groups"]
        for item in group["topics"]
    ]
    assert len(queries) == len(set(queries))
    broad = [
        item
        for group in shipped["groups"]
        for item in group["topics"]
        if item.get("broad")
    ]
    # The 100 niches of the test sample plus broad topics of each area.
    assert len(queries) - len(broad) == 100
    assert len(broad) == 6 * len(shipped["groups"])
    assert shipped["default_limit"] == 250
    with app_with(Provider([]), Crawls()) as client:
        answer = client.get("/api/ingest/topics/presets")
    assert answer.status_code == 200
    assert answer.json() == shipped
