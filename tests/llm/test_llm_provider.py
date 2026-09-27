"""Offline provider contract tests; no live network or paid API requests."""

import asyncio
import json
from copy import deepcopy
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from lctrend.core.config import load_catalog
from lctrend.llm.client import JsonLLM, LLMError, ReplayProvider
from lctrend.llm.contracts import Extraction


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


@pytest.fixture(autouse=True)
def isolated_llm_environment(monkeypatch):
    for name in (
        "LLM_MODEL",
        "LLM_EXTRACT_MODEL",
        "LLM_REVIEW_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LCTREND_CONFIG_DIR",
        "LLM_PROVIDER",
        "LLM_MODEL_LADDER",
        "LLM_CA_BUNDLE_FILE",
        "LCTREND_EXTRACTOR",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_SCOPE",
        "GIGACHAT_BASE_URL",
        "GIGACHAT_AUTH_URL",
        "GIGACHAT_CA_BUNDLE_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def completion(
    content='{"text":"source-backed"}',
    *,
    finish="stop",
    usage=None,
    refusal=None,
):
    return {
        "choices": [
            {
                "message": {"content": content, "refusal": refusal},
                "finish_reason": finish,
            }
        ],
        "usage": usage,
    }


def client(handler, **kwargs):
    return JsonLLM(
        "extract-model",
        base_url="http://127.0.0.1:12345/v1",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def test_stage_models_schema_request_and_safe_audit(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "general")
    monkeypatch.setenv("LLM_EXTRACT_MODEL", "extractor")
    monkeypatch.setenv("LLM_REVIEW_MODEL", "reviewer")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "secret-test-key")
    sent = []

    def respond(request):
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer secret-test-key"
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json=completion(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "untrusted": "raw-secret",
                }
            ),
        )

    provider = JsonLLM.from_environment(transport=httpx.MockTransport(respond))
    assert (
        asyncio.run(
            provider.generate(
                Answer, "Read evidence", {"document_version": "v1"}
            )
        ).text
        == "source-backed"
    )
    asyncio.run(
        provider.generate(
            Answer,
            "Review evidence",
            {"document_version": "v1"},
            stage="review",
        )
    )
    assert [request["model"] for request in sent] == ["extractor", "reviewer"]
    assert all(
        request["response_format"] == {"type": "json_object"}
        for request in sent
    )
    assert sent[0]["max_tokens"] == 4096
    assert '"properties"' in sent[0]["messages"][0]["content"]
    assert provider.calls[0]["tokens"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
    }
    assert provider.calls[0]["estimated_cost_usd"] is None
    audit = json.dumps(list(provider.calls))
    assert "secret-test-key" not in audit and "raw-secret" not in audit
    assert (
        provider.calls[0]["request_sha256"]
        != provider.calls[1]["request_sha256"]
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://llm.example/v1",
        "https://user:pass@llm.example/v1",
        "https://llm.example/v1?key=secret",
        "https://llm.example/v1#secret",
        "file:///tmp/llm",
        "http://127.0.0.1:invalid/v1",
    ],
)
def test_bad_endpoint_rejected_without_exposing_values(url):
    with pytest.raises(LLMError) as failure:
        JsonLLM("m", base_url=url, api_key="secret")
    assert failure.value.code == "configuration"
    assert "secret" not in str(failure.value) and "pass" not in str(
        failure.value
    )


def test_remote_requires_key_and_local_does_not():
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        JsonLLM("m", base_url="https://llm.example/v1")
    JsonLLM("m", base_url="http://localhost:1234/v1")
    JsonLLM("m", base_url="http://[::1]:1234/v1")


@pytest.mark.parametrize(
    "status,retryable",
    [
        (408, True),
        (429, True),
        (500, True),
        (503, True),
        (401, False),
        (403, False),
        (400, False),
        (302, False),
    ],
)
def test_http_error_is_single_attempt_sanitized_and_classified(
    status, retryable
):
    attempted = []

    def respond(request):
        attempted.append(request)
        return httpx.Response(
            status,
            text="raw-response-secret",
            headers={
                "Retry-After": "3.5",
                "Location": "https://other.example",
            },
        )

    provider = client(respond)
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert len(attempted) == 1
    assert failure.value.code == "http_error"
    assert failure.value.retryable is retryable
    assert failure.value.retry_after == (3.5 if retryable else None)
    assert "raw-response-secret" not in str(failure.value)
    assert provider.calls[0]["status"] == "error"


@pytest.mark.parametrize(
    "exception,code",
    [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "transport_error")],
)
def test_transport_errors_are_retryable_but_never_retried(exception, code):
    attempted = []

    def respond(request):
        attempted.append(request)
        raise exception("untrusted-secret-url", request=request)

    provider = client(respond)
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert len(attempted) == 1
    assert failure.value.code == code and failure.value.retryable
    assert "untrusted-secret" not in str(failure.value)
    assert failure.value.__suppress_context__


@pytest.mark.parametrize(
    "data,code",
    [
        (completion(refusal="private-refusal"), "refusal"),
        (completion(finish="length"), "incomplete_response"),
        (completion(finish="content_filter"), "refusal"),
        (completion('{"other":"raw-source-secret"}'), "invalid_schema"),
        (completion("not json raw-source-secret"), "invalid_schema"),
        (completion(""), "invalid_response"),
        ({"choices": []}, "invalid_response"),
        ({"error": {"message": "raw-source-secret"}}, "provider_error"),
    ],
)
def test_invalid_answers_fail_without_fallback(data, code):
    provider = client(lambda _: httpx.Response(200, json=data))
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert failure.value.code == code and not failure.value.retryable
    assert "raw-source-secret" not in str(failure.value)
    assert provider.calls[0]["status"] == "error"


def test_json_and_request_failures_are_explicit():
    provider = client(lambda _: httpx.Response(200, text="not-json-secret"))
    with pytest.raises(LLMError, match="invalid_response"):
        asyncio.run(provider.generate(Answer, "s", {}))
    with pytest.raises(LLMError, match="invalid_request"):
        asyncio.run(provider.generate(Answer, "s", {"number": float("nan")}))
    with pytest.raises(LLMError, match="Stage must"):
        asyncio.run(provider.generate(Answer, "s", {}, stage="search"))


def test_cache_disabled_by_default(tmp_path):
    config = load_catalog("llm")
    config["cache"]["directory"] = str(tmp_path / "disabled")
    attempted = []

    def respond(request):
        attempted.append(request)
        return httpx.Response(200, json=completion())

    provider = client(respond, config=config)
    asyncio.run(provider.generate(Answer, "s", {}))
    asyncio.run(provider.generate(Answer, "s", {}))
    assert len(attempted) == 2
    assert not (tmp_path / "disabled").exists()


def test_success_cache_full_request_key_and_revalidation(tmp_path):
    config = load_catalog("llm")
    config["cache"] = {"enabled": True, "directory": str(tmp_path)}
    attempted = []

    def respond(request):
        attempted.append(request)
        return httpx.Response(
            200,
            json=completion(
                usage={"prompt_tokens": 1, "completion_tokens": 2}
            ),
        )

    provider = client(respond, config=config)
    asyncio.run(
        provider.generate(Answer, "s", {"version": "v1", "chunk_id": "c1"})
    )
    asyncio.run(
        provider.generate(Answer, "s", {"chunk_id": "c1", "version": "v1"})
    )
    assert len(attempted) == 1 and provider.calls[-1]["cache_hit"]
    assert provider.calls[-1]["tokens"] == {}
    assert provider.calls[-1]["cached_response_tokens"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
    }
    assert provider.calls[-1]["estimated_cost_usd"] is None
    asyncio.run(
        provider.generate(Answer, "s", {"version": "v2", "chunk_id": "c1"})
    )
    asyncio.run(
        provider.generate(
            Answer, "changed prompt", {"version": "v1", "chunk_id": "c1"}
        )
    )
    asyncio.run(
        provider.generate(
            Answer, "s", {"version": "v1", "chunk_id": "c1"}, stage="review"
        )
    )
    other = JsonLLM(
        "different-model",
        base_url=provider.base_url,
        config=config,
        transport=httpx.MockTransport(respond),
    )
    asyncio.run(
        other.generate(Answer, "s", {"version": "v1", "chunk_id": "c1"})
    )
    endpoint = JsonLLM(
        "extract-model",
        base_url="http://localhost:12345/v1",
        config=config,
        transport=httpx.MockTransport(respond),
    )
    asyncio.run(
        endpoint.generate(Answer, "s", {"version": "v1", "chunk_id": "c1"})
    )
    assert len(attempted) == 6
    path = tmp_path / (provider.calls[0]["request_sha256"] + ".json")
    cached = json.loads(path.read_text(encoding="utf-8"))
    cached["response"] = {"wrong": "raw-secret"}
    path.write_text(json.dumps(cached), encoding="utf-8")
    with pytest.raises(LLMError, match="invalid_schema"):
        asyncio.run(
            provider.generate(Answer, "s", {"version": "v1", "chunk_id": "c1"})
        )
    assert len(attempted) == 6


def test_only_success_is_cached(tmp_path):
    config = load_catalog("llm")
    config["cache"] = {"enabled": True, "directory": str(tmp_path)}
    provider = client(
        lambda _: httpx.Response(200, json=completion('{"invalid":true}')),
        config=config,
    )
    with pytest.raises(LLMError):
        asyncio.run(provider.generate(Answer, "s", {}))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
@pytest.mark.parametrize("field", ["values", "qualifiers"])
def test_nonfinite_nested_any_in_http_response_never_enters_success_cache(
    tmp_path, nonfinite, field
):
    config = load_catalog("llm")
    config["cache"] = {"enabled": True, "directory": str(tmp_path)}
    response = {
        "entities": [],
        "claims": [
            {
                "claim_id": "candidate",
                "predicate": "reports_limitation",
                "roles": {},
                "evidence": [{"chunk_id": "c1", "quote": "Literal source."}],
            }
        ],
    }
    if field == "values":
        response["claims"][0]["values"] = [
            {"raw": "1 test", "value": nonfinite}
        ]
    else:
        response["claims"][0]["qualifiers"] = {
            "conditions": [{"nested": {"value": nonfinite}}]
        }
    attempted = []

    def respond(request):
        attempted.append(request)
        # json.dumps intentionally retains the forbidden NaN/Infinity tokens;
        # Pydantic accepts them in Any fields unless the provider gates them.
        return httpx.Response(200, json=completion(json.dumps(response)))

    provider = client(respond, config=config)
    for _ in range(2):
        with pytest.raises(LLMError) as failure:
            asyncio.run(provider.generate(Extraction, "s", {}))
        assert failure.value.code == "invalid_schema"
        assert not failure.value.retryable
    assert len(attempted) == 2
    assert not list(tmp_path.iterdir())
    assert all(
        not call["cache_hit"] and call["status"] == "error"
        for call in provider.calls
    )


def test_nonfinite_cached_answer_is_revalidated_and_rejected(tmp_path):
    class NestedAnswer(BaseModel):
        data: dict[str, Any]

    config = load_catalog("llm")
    config["cache"] = {"enabled": True, "directory": str(tmp_path)}
    provider = client(
        lambda _: httpx.Response(
            200, json=completion('{"data":{"values":[1]}}')
        ),
        config=config,
    )
    asyncio.run(provider.generate(NestedAnswer, "s", {}))
    path = tmp_path / (provider.calls[0]["request_sha256"] + ".json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["response"]["data"]["values"] = [{"nested": float("nan")}]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(NestedAnswer, "s", {}))
    assert failure.value.code == "invalid_schema"
    assert (
        provider.calls[-1]["cache_hit"]
        and provider.calls[-1]["status"] == "error"
    )


@pytest.mark.parametrize(
    "nonfinite", [float("nan"), float("inf"), float("-inf")]
)
def test_typed_replay_rejects_nonfinite_any_before_json_normalization(
    nonfinite,
):
    class NestedAnswer(BaseModel):
        data: dict[str, Any]

    typed = NestedAnswer(data={"conditions": [{"nested": nonfinite}]})
    provider = ReplayProvider([typed])
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(NestedAnswer, "s", {}))
    assert failure.value.code == "invalid_schema"
    assert provider.calls[0]["status"] == "error"


def test_cost_is_only_explicit_and_requires_known_cached_rate():
    config = load_catalog("llm")
    config["prices_usd_per_million_tokens"] = {
        "extract-model": {"input": 2, "output": 3}
    }
    provider = client(
        lambda _: httpx.Response(
            200,
            json=completion(
                usage={"prompt_tokens": 100, "completion_tokens": 20}
            ),
        ),
        config=config,
    )
    asyncio.run(provider.generate(Answer, "s", {}))
    assert provider.calls[0]["estimated_cost_usd"] == pytest.approx(0.00026)
    cached_config = deepcopy(config)
    provider = client(
        lambda _: httpx.Response(
            200,
            json=completion(
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 50},
                }
            ),
        ),
        config=cached_config,
    )
    asyncio.run(provider.generate(Answer, "s", {}))
    assert provider.calls[0]["estimated_cost_usd"] is None


def test_replay_is_explicit_validated_sequence_and_no_network(tmp_path):
    path = tmp_path / "recorded.json"
    path.write_text(
        json.dumps(
            {
                "answers": [
                    {
                        "stage": "extract",
                        "schema": "Answer",
                        "response": {"text": "first"},
                    },
                    {"stage": "review", "response": {"text": "second"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = ReplayProvider.from_file(path)
    assert provider.demo is True
    assert asyncio.run(provider.generate(Answer, "s", {})).text == "first"
    assert (
        asyncio.run(provider.generate(Answer, "s", {}, stage="review")).text
        == "second"
    )
    with pytest.raises(LLMError, match="replay_exhausted"):
        asyncio.run(provider.generate(Answer, "s", {}))
    assert len(provider.calls) == 3
    assert all(call["demo"] for call in provider.calls)


@pytest.mark.parametrize(
    "answer,code",
    [
        (
            {"stage": "review", "response": {"text": "wrong"}},
            "replay_stage_mismatch",
        ),
        (
            {"schema": "Different", "response": {"text": "wrong"}},
            "replay_schema_mismatch",
        ),
        ({"text": None}, "invalid_schema"),
    ],
)
def test_replay_rejects_wrong_stage_schema_or_answer(answer, code):
    with pytest.raises(LLMError) as failure:
        asyncio.run(ReplayProvider([answer]).generate(Answer, "s", {}))
    assert failure.value.code == code


def ladder_config(**overrides):
    config = deepcopy(load_catalog("llm"))
    config.update(overrides)
    return config


def test_exhausted_model_falls_down_the_ladder_and_stays_retired():
    sent = []

    def respond(request):
        model = json.loads(request.content)["model"]
        sent.append(model)
        if model == "strong":
            return httpx.Response(402, text="Payment Required raw-secret")
        return httpx.Response(
            200,
            json=completion(
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                }
            ),
        )

    provider = JsonLLM(
        base_url="http://127.0.0.1:1/v1",
        transport=httpx.MockTransport(respond),
        config=ladder_config(model_ladder=["strong", "middle", "weak"]),
    )
    assert provider.models == {"extract": "strong", "review": "strong"}
    assert (
        asyncio.run(provider.generate(Answer, "s", {})).text == "source-backed"
    )
    asyncio.run(provider.generate(Answer, "s", {}, stage="review"))
    assert sent == ["strong", "middle", "middle"]
    assert provider.models == {"extract": "middle", "review": "middle"}
    assert [
        (c["model"], c["status"], c.get("error_code")) for c in provider.calls
    ] == [
        ("strong", "error", "model_exhausted"),
        ("middle", "ok", None),
        ("middle", "ok", None),
    ]
    assert provider.calls[1]["ladder_position"] == 1
    assert [(e["model"], e["event"]) for e in provider.model_events] == [
        ("strong", "model_exhausted")
    ]
    assert "raw-secret" not in json.dumps(
        list(provider.calls) + provider.model_events
    )


def test_all_models_exhausted_is_explicit_and_not_retryable():
    provider = JsonLLM(
        base_url="http://127.0.0.1:1/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(402)),
        config=ladder_config(model_ladder=["a", "b"]),
    )
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert (
        failure.value.code == "models_exhausted"
        and not failure.value.retryable
    )
    assert "a=model_exhausted" in str(
        failure.value
    ) and "b=model_exhausted" in str(failure.value)
    assert len(provider.calls) == 2


def test_pinned_model_degrades_only_to_weaker_ladder_models(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_LADDER", "ultra, max ,pro,lite")
    monkeypatch.setenv("LLM_REVIEW_MODEL", "pro")
    provider = JsonLLM("max", base_url="http://127.0.0.1:1/v1")
    assert provider.ladders == {
        "extract": ["max", "pro", "lite"],
        "review": ["pro", "lite"],
    }
    assert JsonLLM("outside", base_url="http://127.0.0.1:1/v1").ladders[
        "extract"
    ] == ["outside"]


def test_other_http_errors_do_not_move_down_the_ladder():
    provider = JsonLLM(
        base_url="http://127.0.0.1:1/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(429)),
        config=ladder_config(model_ladder=["a", "b"]),
    )
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert failure.value.code == "http_error" and failure.value.retryable
    assert provider.retired == {} and len(provider.calls) == 1


FENCED = "`" * 3 + 'json\n{"text":"source-backed"}\n' + "`" * 3


class GigaChatServer:
    """Offline double of the Sber OAuth, /balance and chat endpoints."""

    def __init__(self, balance=None, statuses=None, content=FENCED):
        self.balance = balance
        self.statuses = dict(statuses or {})
        self.content = content
        self.auth, self.chat, self.balance_requests = [], [], 0
        self.expired = set()

    def __call__(self, request):
        if request.url.host == "ngw.devices.sberbank.ru":
            self.auth.append(request)
            token = f"token-{len(self.auth)}"
            return httpx.Response(
                200, json={"access_token": token, "expires_at": 4102444800000}
            )
        token = request.headers.get("authorization", "")[len("Bearer ") :]
        if token in self.expired:
            return httpx.Response(401)
        if request.url.path == "/v1/balance":
            self.balance_requests += 1
            if self.balance is None:
                return httpx.Response(403)
            return httpx.Response(
                200,
                json={
                    "balance": [
                        {"usage": k, "value": v}
                        for k, v in self.balance.items()
                    ]
                },
            )
        body = json.loads(request.content)
        self.chat.append(body)
        status = self.statuses.get(body["model"], 200)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": self.content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 100,
                    "total_tokens": 1000,
                },
            },
        )


def gigachat(server, **kwargs):
    return JsonLLM(
        provider="gigachat",
        api_key="basic-secret",
        transport=httpx.MockTransport(server),
        **kwargs,
    )


def test_gigachat_oauth_schema_request_and_default_ladder():
    server = GigaChatServer(balance=None)
    provider = gigachat(server)
    assert provider.base_url == "https://api.giga.chat/v1"
    assert provider.ladders["extract"] == [
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
        "GigaChat-2-Pro",
        "GigaChat-2",
    ]
    assert (
        asyncio.run(provider.generate(Answer, "Read evidence", {"v": 1})).text
        == "source-backed"
    )
    asyncio.run(provider.generate(Answer, "Review", {"v": 1}, stage="review"))
    auth = server.auth[0]
    assert len(server.auth) == 1, "access token is reused until it expires"
    assert auth.headers["authorization"] == "Basic basic-secret"
    assert auth.headers["rquid"] and auth.content == b"scope=GIGACHAT_API_PERS"
    request = server.chat[0]
    assert request["model"] == "GigaChat-3-Ultra"
    assert request["max_tokens"] == 8192
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["strict"] is True
    assert (
        request["response_format"]["schema"]["properties"]["text"]["type"]
        == "string"
    )
    assert 0 < request["temperature"] < 0.01
    assert provider.calls[0]["provider"] == "gigachat"
    assert (
        provider.balance_status == "http_403" and not provider.balance_enabled
    )
    assert "basic-secret" not in json.dumps(
        list(provider.calls)
    ) and "token-1" not in json.dumps(list(provider.calls))


def test_gigachat_schema_has_no_unresolved_local_references():
    server = GigaChatServer(content=json.dumps({"entities": [], "claims": []}))
    asyncio.run(gigachat(server).generate(Extraction, "s", {}))
    schema = json.dumps(server.chat[0]["response_format"]["schema"])
    assert "$ref" not in schema and "$defs" not in schema
    assert server.chat[0]["response_format"]["schema"]["required"] == [
        "entities", "claims", "context_requests"
    ]


def test_gigachat_balance_skips_low_models_and_402_moves_on():
    server = GigaChatServer(
        balance={
            "GigaChat-3-Ultra": 10,
            "GigaChat-2-Max": 5_000_000,
            "GigaChat-Pro": 70_000,
        },
        statuses={"GigaChat-2-Max": 402},
    )
    provider = gigachat(server)
    asyncio.run(provider.generate(Answer, "s", {}))
    assert [body["model"] for body in server.chat] == [
        "GigaChat-2-Max",
        "GigaChat-2-Pro",
    ]
    assert [(e["model"], e["event"]) for e in provider.model_events] == [
        ("GigaChat-3-Ultra", "low_balance"),
        ("GigaChat-2-Max", "model_exhausted"),
    ]
    assert provider.balance["GigaChat-2-Pro"] == 69_000
    asyncio.run(provider.generate(Answer, "s", {}))
    assert server.chat[-1]["model"] == "GigaChat-2-Pro"
    assert server.balance_requests == 1, (
        "balance is re-read only after refresh_seconds"
    )


def test_gigachat_local_balance_estimate_retires_model_below_reserve():
    # Reserve is 20 000 + 8 192 output tokens; one 1 000-token answer
    # crosses it.
    server = GigaChatServer(
        balance={"GigaChat-3-Ultra": 29_000, "GigaChat-2-Max": 1_000_000}
    )
    provider = gigachat(server)
    asyncio.run(provider.generate(Answer, "s", {}))
    asyncio.run(provider.generate(Answer, "s", {}))
    assert [body["model"] for body in server.chat] == [
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
    ]
    assert provider.retired == {"GigaChat-3-Ultra": "low_balance"}


def test_gigachat_ultra_forbidden_for_paid_account_falls_through():
    server = GigaChatServer(statuses={"GigaChat-3-Ultra": 403})
    provider = gigachat(server)
    asyncio.run(provider.generate(Answer, "s", {}))
    assert [body["model"] for body in server.chat] == [
        "GigaChat-3-Ultra",
        "GigaChat-2-Max",
    ]
    assert provider.retired == {"GigaChat-3-Ultra": "model_unavailable"}


def test_gigachat_expired_token_is_renewed_once():
    server = GigaChatServer()
    provider = gigachat(server)
    asyncio.run(provider.generate(Answer, "s", {}))
    server.expired.add("token-1")
    asyncio.run(provider.generate(Answer, "s", {}))
    assert len(server.auth) == 2 and len(provider.calls) == 2
    assert provider.calls[-1]["status"] == "ok"


def test_gigachat_configuration_uses_its_own_credentials(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LLM_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    with pytest.raises(LLMError, match="GIGACHAT_CREDENTIALS"):
        JsonLLM.from_environment()
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "basic-secret")
    monkeypatch.setenv(
        "GIGACHAT_CA_BUNDLE_FILE", str(tmp_path / "missing.crt")
    )
    with pytest.raises(LLMError, match="CA bundle"):
        JsonLLM.from_environment()
    bundle = tmp_path / "russian_trusted_root_ca.crt"
    bundle.write_text("fixture", encoding="utf-8")
    monkeypatch.setenv("GIGACHAT_CA_BUNDLE_FILE", str(bundle))
    provider = JsonLLM.from_environment()
    assert (
        provider.base_url == "https://api.giga.chat/v1"
        and provider.verify == str(bundle)
    )
    with pytest.raises(LLMError, match="LLM_PROVIDER"):
        JsonLLM(provider="other", api_key="x")


@pytest.mark.parametrize("status", [401, 500])
def test_gigachat_auth_failure_is_sanitized(status):
    def respond(request):
        return httpx.Response(status, text="raw-auth-secret")

    with pytest.raises(LLMError) as failure:
        asyncio.run(gigachat(respond).generate(Answer, "s", {}))
    assert failure.value.code == "auth_error"
    assert failure.value.retryable is (status == 500)
    assert "raw-auth-secret" not in str(failure.value)


def test_gigachat_embeddings_share_oauth_and_keep_input_order():
    requests = []

    def respond(request):
        if request.url.host == "ngw.devices.sberbank.ru":
            return httpx.Response(
                200,
                json={"access_token": "token-1", "expires_at": 4102444800000},
            )
        body = json.loads(request.content)
        requests.append((request.url.path, request.headers, body))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {
                        "object": "embedding",
                        "index": 1,
                        "embedding": [0.0, 2.0],
                        "usage": {"prompt_tokens": 3},
                    },
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [1.0, 0.0],
                        "usage": {"prompt_tokens": 4},
                    },
                ],
            },
        )

    provider = gigachat(respond)
    vectors = asyncio.run(provider.embed(["a", "b"], "EmbeddingsGigaR"))
    assert vectors == [[1.0, 0.0], [0.0, 2.0]]
    path, headers, body = requests[0]
    assert path == "/v1/embeddings"
    assert headers["authorization"] == "Bearer token-1"
    assert body == {"model": "EmbeddingsGigaR", "input": ["a", "b"]}
    assert provider.calls[-1]["stage"] == "embed"
    assert provider.calls[-1]["tokens"] == {"prompt_tokens": 7}


def test_gigachat_embeddings_reject_mismatched_response():
    def respond(request):
        if request.url.host == "ngw.devices.sberbank.ru":
            return httpx.Response(
                200,
                json={"access_token": "t", "expires_at": 4102444800000},
            )
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

    with pytest.raises(LLMError) as failure:
        asyncio.run(gigachat(respond).embed(["a", "b"], "EmbeddingsGigaR"))
    assert failure.value.code == "invalid_response"
