"""Key pool: several GigaChat keys serve requests at once, offline."""

import asyncio
import base64
import json
from copy import deepcopy

import httpx
import pytest

from lctrend.core.config import load_catalog
from lctrend.llm.client import JsonLLM, KeyPool, LLMError, load_keys
from tests.llm.test_llm_provider import Answer


@pytest.fixture(autouse=True)
def unpinned(monkeypatch):
    for name in (
        "LLM_MODEL",
        "LLM_MODEL_LADDER",
        "LLM_EXTRACT_MODEL",
        "LLM_REVIEW_MODEL",
        "LLM_MAX_CONCURRENCY",
        "GIGACHAT_CREDENTIALS",
        "GIGACHAT_SCOPE",
        "GIGACHAT_BASE_URL",
        "GIGACHAT_CA_BUNDLE_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def basic(client_id, secret="secret"):
    return base64.b64encode(f"{client_id}:{secret}".encode()).decode()


class Accounts:
    """Sber OAuth and chat double that tracks requests per account."""

    def __init__(
        self,
        exhausted=(),
        delay=0.05,
        limited=(),
        revoked=(),
        forbidden=(),
        no_embeddings=(),
    ):
        self.no_embeddings = set(no_embeddings)
        self.exhausted = set(exhausted)
        self.limited = set(limited)
        self.revoked = set(revoked)
        self.forbidden = set(forbidden)
        self.delay = delay
        self.tokens = {}
        self.chat = []
        self.embeddings = []
        self.active = {}
        self.peak = {}
        self.peak_total = 0

    async def __call__(self, request):
        if request.url.host == "ngw.devices.sberbank.ru":
            account = base64.b64decode(
                request.headers["authorization"][len("Basic ") :]
            ).decode().split(":")[0]
            token = f"token-{account}"
            self.tokens[token] = account
            return httpx.Response(
                200, json={"access_token": token, "expires_at": 4102444800000}
            )
        account = self.tokens[request.headers["authorization"][7:]]
        if request.url.path == "/v1/balance":
            return httpx.Response(403)
        self.active[account] = self.active.get(account, 0) + 1
        self.peak[account] = max(
            self.peak.get(account, 0), self.active[account]
        )
        self.peak_total = max(self.peak_total, sum(self.active.values()))
        try:
            await asyncio.sleep(self.delay)
            body = json.loads(request.content)
            if account in self.limited:
                self.chat.append((account, "429"))
                return httpx.Response(429)
            if account in self.revoked:
                self.chat.append((account, "401"))
                return httpx.Response(401)
            if request.url.path == "/v1/embeddings":
                if account in self.no_embeddings:
                    return httpx.Response(402)
                self.embeddings.append(account)
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"index": i, "embedding": [1.0, float(i)]}
                            for i in range(len(body["input"]))
                        ]
                    },
                )
            self.chat.append((account, body["model"]))
            if account in self.exhausted:
                return httpx.Response(402)
            if account in self.forbidden:
                return httpx.Response(403)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"text":"ok"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 10},
                },
            )
        finally:
            self.active[account] -= 1


def write_keys(tmp_path, keys):
    path = tmp_path / "gigachat-keys.json"
    path.write_text(json.dumps({"keys": keys}), encoding="utf-8")
    return path


def pool(tmp_path, server, keys):
    config = deepcopy(load_catalog("llm"))
    config["gigachat"].pop("model_routes", None)
    return KeyPool.from_file(
        write_keys(tmp_path, keys),
        transport=httpx.MockTransport(server),
        config=config,
    )


def test_keys_file_accepts_client_id_and_secret_or_authorization_key(
    tmp_path,
):
    keys = load_keys(
        write_keys(
            tmp_path,
            [
                {"client_id": "aaaa1111-x", "client_secret": "s1"},
                {"credentials": basic("bbbb2222-y"), "workers": 2},
                {"client_id": "cccc", "client_secret": "s3", "enabled": False},
                {
                    "name": "corp",
                    "credentials": basic("dddd"),
                    "scope": "GIGACHAT_API_CORP",
                },
            ],
        )
    )
    assert [(k["name"], k["workers"], k["scope"]) for k in keys] == [
        ("aaaa1111", 1, None),
        ("bbbb2222", 2, None),
        ("corp", 1, "GIGACHAT_API_CORP"),
    ]
    assert keys[0]["credentials"] == basic("aaaa1111-x", "s1")


@pytest.mark.parametrize(
    "keys",
    [
        [{"client_id": "only-id"}],
        [
            {"credentials": basic("a", "hidden")},
            {"client_id": "a", "client_secret": "hidden"},
        ],
        [{"credentials": basic("a"), "workers": 0}],
        [{"credentials": basic("a"), "enabled": False}],
        {"keys": "not-a-list"},
    ],
)
def test_bad_keys_file_is_a_configuration_error_without_secrets(
    tmp_path, keys
):
    path = tmp_path / "keys.json"
    path.write_text(
        json.dumps(keys if isinstance(keys, dict) else {"keys": keys}),
        encoding="utf-8",
    )
    with pytest.raises(LLMError) as failure:
        load_keys(path)
    assert failure.value.code == "configuration"
    assert "hidden" not in str(failure.value)
    assert basic("a", "hidden") not in str(failure.value)


def test_each_key_runs_its_own_workers_at_once(tmp_path):
    server = Accounts()
    provider = pool(
        tmp_path,
        server,
        [
            {"credentials": basic("one")},
            {"credentials": basic("two"), "workers": 2},
        ],
    )
    assert provider.max_concurrency == 3

    async def burst():
        await asyncio.gather(
            *(provider.generate(Answer, "s", {"n": n}) for n in range(9))
        )

    asyncio.run(burst())
    assert server.peak == {"one": 1, "two": 2}
    assert server.peak_total == 3
    assert len(server.chat) == 9
    # Audits name the key, never its secret.
    assert {call["key"] for call in provider.calls} == {"one", "two"}
    assert basic("one") not in json.dumps(list(provider.calls))


def test_key_out_of_tokens_passes_requests_to_the_others(tmp_path):
    server = Accounts(exhausted={"one"})
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    for n in range(3):
        answer = asyncio.run(provider.generate(Answer, "s", {"n": n}))
        assert answer.text == "ok"
    tried_on_one = [model for key, model in server.chat if key == "one"]
    # Key one walks its ladder once, then is never asked again.
    assert tried_on_one == provider.ladders["extract"]
    assert [a for a, _ in server.chat].count("two") == 3
    assert {event["key"] for event in provider.model_events} == {"one"}
    assert provider.models["extract"] == provider.ladders["extract"][0]


def test_every_key_out_of_tokens_is_explicit(tmp_path):
    server = Accounts(exhausted={"one", "two"})
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert failure.value.code == "models_exhausted"
    assert provider.models["extract"] is None


def test_embeddings_use_free_keys_of_the_pool(tmp_path):
    server = Accounts()
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )

    async def burst():
        return await asyncio.gather(
            *(provider.embed(["a", "b"], "EmbeddingsGigaR") for _ in range(4))
        )

    vectors = asyncio.run(burst())
    assert vectors[0] == [[1.0, 0.0], [1.0, 1.0]]
    assert sorted(server.embeddings) == ["one", "one", "two", "two"]
    assert server.peak_total == 2


def test_environment_builds_a_pool_only_for_gigachat(tmp_path, monkeypatch):
    path = write_keys(tmp_path, [{"credentials": basic("one")}])
    monkeypatch.setenv("GIGACHAT_KEYS_FILE", str(path))
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    provider = JsonLLM.from_environment()
    assert isinstance(provider, KeyPool)
    assert [member.key_name for member in provider.members] == ["one"]
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert isinstance(JsonLLM.from_environment(), JsonLLM)


def test_keys_follow_the_studio_fields_and_catch_mixups(tmp_path):
    keys = load_keys(
        write_keys(
            tmp_path,
            [
                {
                    "client_id": "aaaa1111",
                    "auth_key": basic("aaaa1111"),
                    "scope": "GIGACHAT_API_PERS",
                },
                # The Authorization key pasted into client_secret.
                {"client_id": "", "client_secret": basic("bbbb2222")},
            ],
        )
    )
    assert [(k["name"], k["credentials"]) for k in keys] == [
        ("aaaa1111", basic("aaaa1111")),
        ("bbbb2222", basic("bbbb2222")),
    ]
    for wrong in (
        {"client_id": "aaaa1111", "auth_key": basic("other")},
        {"auth_key": "not base64 at all"},
    ):
        with pytest.raises(LLMError) as failure:
            load_keys(write_keys(tmp_path, [wrong]))
        assert failure.value.code == "configuration"


def test_stats_show_queue_and_request_time_per_key(tmp_path):
    from lctrend.llm.stats import STATS

    STATS.reset()
    server = Accounts(delay=0.05)
    provider = pool(tmp_path, server, [{"credentials": basic("one")}])

    async def burst():
        await asyncio.gather(
            *(provider.generate(Answer, "s", {"n": n}) for n in range(3))
        )

    asyncio.run(burst())
    # One slot: the later requests waited for the earlier ones.
    # Windows timers wake a sleep a few ms early.
    assert max(call["queue_ms"] for call in provider.calls) >= 30
    snapshot = STATS.snapshot()
    key = snapshot["keys"]["one"]
    assert key["capacity"] == 1 and key["in_flight"] == 0
    assert key["totals"]["calls"] == 3 and key["totals"]["errors"] == 0
    assert key["recent"]["queue_ms_p95"] >= 30
    assert snapshot["stages"]["extract"]["request_ms_p50"] >= 30
    assert snapshot["waiting_requests"] == 0
    assert basic("one") not in json.dumps(snapshot)


def test_a_rate_limited_key_rests_and_the_others_serve(tmp_path):
    server = Accounts(limited={"one"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    for n in range(4):
        answer = asyncio.run(provider.generate(Answer, "s", {"n": n}))
        assert answer.text == "ok"
    asked_one = [key for key, _ in server.chat if key == "one"]
    # Key one answers 429 once, then rests while two serves the rest.
    assert asked_one == ["one"]
    assert [key for key, _ in server.chat].count("two") == 4
    vectors = asyncio.run(provider.embed(["a"], "EmbeddingsGigaR"))
    assert vectors == [[1.0, 0.0]] and server.embeddings == ["two"]


def test_every_key_rate_limited_is_the_callers_retry(tmp_path):
    server = Accounts(limited={"one", "two"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.generate(Answer, "s", {}))
    assert failure.value.retryable
    assert sorted(key for key, _ in server.chat) == ["one", "two"]


def test_a_refused_key_leaves_the_rotation(tmp_path):
    server = Accounts(revoked={"one"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    for n in range(4):
        answer = asyncio.run(provider.generate(Answer, "s", {"n": n}))
        assert answer.text == "ok"
    # One request (and its single token renewal) finds the key refused;
    # the key then rests for the run.
    assert [key for key, _ in server.chat].count("one") == 2
    assert [key for key, _ in server.chat].count("two") == 4


def test_rate_limit_is_read_from_the_status_not_the_message():
    from lctrend.llm.client import rate_limited, rejected_key

    assert rate_limited(LLMError("http_error", "anything", True, status=429))
    assert not rate_limited(LLMError("http_error", "HTTP 429 in text", True))
    assert rejected_key(LLMError("http_error", "x", status=401))
    # A 403 is a model outside the key's plan, not a refused key.
    assert not rejected_key(LLMError("http_error", "x", status=403))
    assert rejected_key(LLMError("auth_error", "no token"))
    assert not rejected_key(LLMError("auth_error", "busy", True, status=429))


def test_a_pool_of_refused_keys_fails_at_once(tmp_path):
    server = Accounts(revoked={"one", "two"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )

    async def twice():
        for _ in range(2):
            with pytest.raises(LLMError):
                await asyncio.wait_for(
                    provider.generate(Answer, "s", {}), timeout=5
                )

    asyncio.run(twice())
    with pytest.raises(LLMError) as failure:
        asyncio.run(asyncio.wait_for(provider.embed(["a"], "E"), timeout=5))
    assert failure.value.code == "auth_error"


def test_a_forbidden_model_retires_on_its_key_only(tmp_path):
    server = Accounts(forbidden={"one"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    for n in range(3):
        answer = asyncio.run(provider.generate(Answer, "s", {"n": n}))
        assert answer.text == "ok"
    tried_on_one = [model for key, model in server.chat if key == "one"]
    # 403 walks key one's ladder once, like a model it cannot use.
    assert tried_on_one == provider.ladders["extract"]


def test_embeddings_find_the_key_that_has_them_and_chat_leaves_it(tmp_path):
    # As the live pool: only "main" has an embeddings package.
    server = Accounts(no_embeddings={"two", "three"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [
            {"credentials": basic("main")},
            {"credentials": basic("two")},
            {"credentials": basic("three")},
        ],
    )
    for _ in range(4):
        vectors = asyncio.run(provider.embed(["a"], "EmbeddingsGigaR"))
        assert vectors == [[1.0, 0.0]]
    assert server.embeddings == ["main"] * 4
    # Chat now leaves the embedding key to embeddings.
    for n in range(4):
        asyncio.run(provider.generate(Answer, "s", {"n": n}))
    assert "main" not in [key for key, _ in server.chat]


def test_no_key_with_embeddings_is_explicit(tmp_path):
    server = Accounts(no_embeddings={"one", "two"}, delay=0)
    provider = pool(
        tmp_path,
        server,
        [{"credentials": basic("one")}, {"credentials": basic("two")}],
    )
    with pytest.raises(LLMError) as failure:
        asyncio.run(provider.embed(["a"], "EmbeddingsGigaR"))
    assert failure.value.code == "models_exhausted"


def test_chat_pool_learns_what_the_embedding_pool_found(tmp_path):
    # The semantic layer and the extraction build separate pools.
    server = Accounts(no_embeddings={"two"}, delay=0)
    keys = [{"credentials": basic("main")}, {"credentials": basic("two")}]
    embedding_pool = pool(tmp_path, server, keys)
    chat_pool = pool(tmp_path, server, keys)
    for _ in range(2):
        asyncio.run(embedding_pool.embed(["a"], "EmbeddingsGigaR"))
    for n in range(3):
        asyncio.run(chat_pool.generate(Answer, "s", {"n": n}))
    assert [key for key, _ in server.chat] == ["two"] * 3
    # The chat pool never asks "two" for embeddings again.
    asyncio.run(chat_pool.embed(["a"], "EmbeddingsGigaR"))
    assert server.embeddings == ["main"] * 3


def test_a_key_rests_for_every_pool_of_the_process(tmp_path):
    # Jobs and the semantic layer build separate pools from one keys file:
    # a 429 on one pool must keep the others off that key too.
    server = Accounts(limited={"one"}, delay=0)
    keys = [{"credentials": basic("one")}, {"credentials": basic("two")}]
    first, second = pool(tmp_path, server, keys), pool(tmp_path, server, keys)
    asyncio.run(first.generate(Answer, "s", {"n": 0}))
    for n in range(3):
        asyncio.run(second.generate(Answer, "s", {"n": n}))
    assert [key for key, _ in server.chat].count("one") == 1


def test_a_key_keeps_one_limit_whoever_builds_its_client(tmp_path):
    # A second client of the same key asking for more requests at once
    # shares the key's limit instead of opening a second gate.
    server = Accounts(delay=0.05)
    config = deepcopy(load_catalog("llm"))
    config["gigachat"].pop("model_routes", None)
    options = {
        "api_key": basic("one"),
        "provider": "gigachat",
        "scope": "GIGACHAT_API_PERS",
        "transport": httpx.MockTransport(server),
        "config": config,
    }
    narrow = JsonLLM(max_concurrency=1, **options)
    wide = JsonLLM(max_concurrency=4, **options)

    async def burst():
        await asyncio.gather(
            *(
                client.generate(Answer, "s", {"n": n})
                for n, client in enumerate([narrow, wide] * 3)
            )
        )

    asyncio.run(burst())
    assert server.peak == {"one": 1}


def test_documents_at_once_never_exceed_the_keys(tmp_path):
    from lctrend.llm.client import document_workers

    provider = pool(
        tmp_path,
        Accounts(),
        [{"credentials": basic(name)} for name in ("a", "b", "c", "d")],
    )
    assert document_workers(6, provider) == 4
    assert document_workers(2, provider) == 2
    # No model calls (mode "none"): the requested workers run.
    assert document_workers(6, None) == 6
