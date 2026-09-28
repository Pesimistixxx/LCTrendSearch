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


def test_bundled_gigachat_routes_start_with_ultra_and_fall_back():
    # 2-Max broke the strict review and topic schemas on live calls, so
    # every bundled route starts with 3-Ultra; 2-Max is the token fallback.
    from tests.llm.test_llm_provider import GigaChatServer, gigachat

    server = GigaChatServer(balance=None, statuses={"GigaChat-3-Ultra": 402})
    provider = gigachat(server, routes=True)
    asyncio.run(provider.generate(Answer, "s", {"abstract": "short"}))
    asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    assert [body["model"] for body in server.chat] == [
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
        "GigaChat-2-Max",
    ]
    assert {provider.ladders[key][0] for key in provider.ladders} == {
        "GigaChat-3-Ultra"
    }


def test_schema_free_models_get_no_response_format_and_a_session():
    from tests.llm.test_llm_provider import GigaChatServer, gigachat

    server = GigaChatServer(balance=None)
    headers = []
    original = server.__call__

    def record(request):
        if request.url.path.endswith("/chat/completions"):
            headers.append(request.headers.get("x-session-id"))
        return original(request)

    config = deepcopy(load_catalog("llm"))
    provider = gigachat(record, config=config)
    provider.ladders["review"] = ["GigaChat-2-Max"]
    asyncio.run(provider.generate(Answer, "s", {}))
    asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    ultra, lite = server.chat
    # Constrained, 3-Ultra indents its JSON: 2-5x the output tokens (live,
    # 2026-09-29); unconstrained it writes the compact answer.
    assert ultra["model"] == "GigaChat-3-Ultra"
    assert "response_format" not in ultra
    # Long strict answers of GigaChat-2 models broke the JSON (live, 2026).
    assert lite["model"] == "GigaChat-2-Max" and "response_format" not in lite
    # One cached prompt prefix per model and system message.
    assert all(headers) and headers[0] != headers[1]


def test_broken_unconstrained_answer_is_resent_under_the_schema():
    from tests.llm.test_llm_provider import GigaChatServer, gigachat

    server = GigaChatServer(balance=None)
    answers = iter(['{"text": "cut', '{"text":"source-backed"}'])
    original = server.__call__

    def respond(request):
        if request.url.path.endswith("/chat/completions"):
            server.content = next(answers)
        return original(request)

    provider = gigachat(respond, config=deepcopy(load_catalog("llm")))
    result = asyncio.run(provider.generate(Answer, "s", {}))
    assert result.text == "source-backed"
    free, strict = server.chat
    assert "response_format" not in free
    assert strict["response_format"]["type"] == "json_schema"
    assert [call.get("constrained") for call in provider.calls] == [
        None,
        True,
    ]


def test_broken_answer_of_a_model_without_fallback_is_not_resent():
    from tests.llm.test_llm_provider import GigaChatServer, gigachat

    server = GigaChatServer(balance=None, content='{"text": "cut')
    provider = gigachat(server, config=deepcopy(load_catalog("llm")))
    provider.ladders = {
        route: ["GigaChat-2-Max"] for route in provider.ladders
    }
    with pytest.raises(LLMError, match="invalid_schema"):
        asyncio.run(provider.generate(Answer, "s", {}))
    assert len(server.chat) == 1


def test_route_naming_an_unknown_model_is_rejected():
    catalogs = {name: deepcopy(load_catalog(name)) for name in CATALOG_NAMES}
    catalogs["llm"]["gigachat"]["model_routes"]["review"] = ["GigaChat-9"]
    with pytest.raises(Exception, match="model_routes.review"):
        validate_catalogs(catalogs)
