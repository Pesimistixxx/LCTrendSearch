"""Model routes: calls spread over models by task, with token fallback."""

import asyncio
import json
from copy import deepcopy

import httpx
import pytest

from lctrend.core.catalog_validation import CATALOG_NAMES, validate_catalogs
from lctrend.core.config import load_catalog
from lctrend.llm.client import JsonLLM, LLMError
from tests.llm.test_llm_provider import Answer, completion, ladder_config


@pytest.fixture(autouse=True)
def unpinned(monkeypatch):
    for name in (
        "LLM_MODEL",
        "LLM_MODEL_LADDER",
        "LLM_EXTRACT_MODEL",
        "LLM_REVIEW_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def routed_config(**routes):
    return ladder_config(
        model_ladder=["ultra", "max", "pro", "lite"],
        model_routes={
            "short_packet_chars": 50,
            "extract": ["ultra", "max"],
            "extract_short": ["max", "pro"],
            "review": ["max"],
            **routes,
        },
    )


def provider_with(respond):
    return JsonLLM(
        base_url="http://127.0.0.1:1/v1",
        transport=httpx.MockTransport(respond),
        config=routed_config(),
    )


def test_routes_send_short_packets_and_reviews_to_lighter_models():
    sent = []

    def respond(request):
        sent.append(json.loads(request.content)["model"])
        return httpx.Response(200, json=completion())

    provider = provider_with(respond)
    # Every route ends with the rest of the ladder: no dead ends.
    assert provider.ladders == {
        "extract": ["ultra", "max", "pro", "lite"],
        "extract_short": ["max", "pro", "ultra", "lite"],
        "review": ["max", "ultra", "pro", "lite"],
    }
    asyncio.run(provider.generate(Answer, "s", {"text": "abstract"}))
    asyncio.run(provider.generate(Answer, "s", {"text": "x" * 200}))
    asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    assert sent == ["max", "ultra", "max"]
    assert [call["route"] for call in provider.calls] == [
        "extract_short",
        "extract",
        "review",
    ]


def test_model_out_of_tokens_is_skipped_on_every_route():
    sent = []

    def respond(request):
        model = json.loads(request.content)["model"]
        sent.append(model)
        if model == "max":
            return httpx.Response(402)
        return httpx.Response(200, json=completion())

    provider = provider_with(respond)
    asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    asyncio.run(provider.generate(Answer, "s", {"t": "short"}))
    # review: max (402) -> ultra; the short extraction skips retired max.
    assert sent == ["max", "ultra", "pro"]
    assert provider.models["extract_short"] == "pro"


def test_every_model_out_of_tokens_is_explicit():
    provider = provider_with(lambda _: httpx.Response(402))
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    assert failure.value.code == "models_exhausted"
    assert len(provider.calls) == 4


def test_a_pinned_model_disables_routes(monkeypatch):
    monkeypatch.setenv("LLM_EXTRACT_MODEL", "ultra")
    provider = JsonLLM(
        base_url="http://127.0.0.1:1/v1", config=routed_config()
    )
    assert "extract_short" not in provider.ladders
    assert provider.ladders["extract"][0] == "ultra"


def test_bundled_gigachat_routes_abstracts_and_reviews_to_max():
    from tests.llm.test_llm_provider import GigaChatServer, gigachat

    server = GigaChatServer(balance=None)
    provider = gigachat(server, routes=True)
    asyncio.run(provider.generate(Answer, "s", {"abstract": "short"}))
    asyncio.run(provider.generate(Answer, "s", {"packet": "x" * 20000}))
    asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    assert [body["model"] for body in server.chat] == [
        "GigaChat-2-Max",
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
    ]
    # The output limit follows the stage, not the route.
    assert server.chat[0]["max_tokens"] == server.chat[1]["max_tokens"]


def test_route_naming_an_unknown_model_is_rejected():
    catalogs = {name: deepcopy(load_catalog(name)) for name in CATALOG_NAMES}
    catalogs["llm"]["gigachat"]["model_routes"]["review"] = ["GigaChat-9"]
    with pytest.raises(Exception, match="model_routes.review"):
        validate_catalogs(catalogs)
