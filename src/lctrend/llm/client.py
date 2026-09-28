"""One synchronous, schema-validated LLM request or explicit offline replay.

The coordinator owns retries, context tools and total call budgets. This
module never retries a failed answer, follows redirects or substitutes
fabricated answers. It only moves a request down the configured model ladder
when a model has no tokens left or is not available to the account.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import threading
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, perf_counter
from typing import Any, Dict, Optional, Protocol, Tuple, Type, TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from ..core.config import load_catalog

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
STAGES = frozenset(("extract", "review"))
# Ladder key of short extraction packets (abstracts) in llm.json
# model_routes; the stage and its output limit stay "extract".
SHORT_EXTRACT_ROUTE = "extract_short"
# Calls of the document being processed. One provider serves concurrent
# documents, so each document task collects its own call audit here.
CALL_LOG: ContextVar[Optional[list]] = ContextVar("llm_call_log", default=None)
# A long crawl keeps one provider alive; its own history stays bounded.
MAX_RETAINED_CALLS = 2000


# One request limit per provider endpoint for the whole process: the CLI,
# the web job loop, the resolver's own client and embedding threads all
# count against LLM_MAX_CONCURRENCY together (B-10).
_PROCESS_LIMITS: Dict[Tuple[str, str, int], threading.BoundedSemaphore] = {}
_PROCESS_LIMITS_LOCK = threading.Lock()


def _process_limit(
    provider: str, base_url: str, limit: int
) -> threading.BoundedSemaphore:
    with _PROCESS_LIMITS_LOCK:
        key = (provider, base_url, limit)
        if key not in _PROCESS_LIMITS:
            _PROCESS_LIMITS[key] = threading.BoundedSemaphore(limit)
        return _PROCESS_LIMITS[key]


class _RequestSlot:
    """Hold one process-wide request slot without blocking the event loop.

    A thread semaphore works across loops and threads; polling it keeps a
    cancelled waiter from ever owning a slot it cannot release.
    """

    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self.semaphore = semaphore

    async def __aenter__(self) -> None:
        while not self.semaphore.acquire(blocking=False):
            await asyncio.sleep(0.01)

    async def __aexit__(self, *_: Any) -> None:
        self.semaphore.release()


def _record(calls: Any, call: Dict[str, Any]) -> None:
    calls.append(call)
    log = CALL_LOG.get()
    if log is not None:
        log.append(call)


class Provider(Protocol):
    async def generate(
        self,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        *,
        stage: str = "extract",
    ) -> T:
        """Return one response validated against the requested schema."""
        ...


def _log_call(call: Dict[str, Any]) -> None:
    """Log the call summary only: prompts and answers may hold source text."""
    logger.debug(
        "LLM call provider=%s stage=%s model=%s status=%s code=%s "
        "cache_hit=%s tokens=%s duration_ms=%s",
        call.get("provider"),
        call.get("stage"),
        call.get("model"),
        call.get("status"),
        call.get("error_code"),
        call.get("cache_hit"),
        call.get("tokens"),
        call.get("duration_ms"),
    )


class LLMError(RuntimeError):
    """Safe machine-readable failure without raw HTTP bodies or secrets."""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
        retry_after: Optional[float] = None,
    ):
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        super().__init__(f"{code}: {message}")


def _finite_numbers(value: Any) -> None:
    """Any-typed nested fields must not hide JSON-invalid numeric values.

    Pydantic can parse NaN/Infinity in JSON and serialize them as null.
    Validate the Python values first, before a cache or replay can normalize
    the answer.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError("Nonfinite JSON number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_numbers(key)
            _finite_numbers(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _finite_numbers(item)


def _validate(schema: Type[T], answer: Any) -> T:
    from .contracts import LocalModel, validate_items

    try:
        if isinstance(answer, BaseModel):
            answer = answer.model_dump(mode="python")
        if isinstance(answer, str):
            answer = json.loads(answer)
        try:
            result = schema.model_validate(answer)
        except ValidationError:
            if not issubclass(schema, LocalModel):
                raise
            # One bad element must not cost the whole packet (B-5).
            result = validate_items(schema, answer)
        _finite_numbers(result.model_dump(mode="python"))
        return result
    except (ValidationError, ValueError, TypeError):
        # Suppress the exception context: validation details can contain source
        # text or a provider's untrusted response.
        raise LLMError(
            "invalid_schema", f"Response is not valid {schema.__name__} JSON"
        ) from None


def _json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (ValueError, TypeError):
        raise LLMError(
            "invalid_request",
            "Request must contain serializable finite JSON values",
        ) from None


def _hash(value: Any) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()


def _positive(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise LLMError(
            "configuration", f"{name} must be a positive finite number"
        )
    return float(value)


def _stage(stage: str) -> None:
    if stage not in STAGES:
        raise LLMError("configuration", "Stage must be extract or review")


def _tokens(usage: Any) -> Dict[str, int]:
    """Keep only recognized numeric usage fields, never response text."""
    if not isinstance(usage, dict):
        return {}
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        ):
            result[key] = value
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if (
            isinstance(cached, int)
            and not isinstance(cached, bool)
            and cached >= 0
        ):
            result["cached_tokens"] = cached
    elif (
        isinstance(usage.get("cached_tokens"), int)
        and not isinstance(usage["cached_tokens"], bool)
        and usage["cached_tokens"] >= 0
    ):
        # Success cache stores this already sanitized flat usage mapping.
        result["cached_tokens"] = usage["cached_tokens"]
    return result


def _retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            moment = parsedate_to_datetime(value)
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            seconds = (moment - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Self-contained schema for providers that do not resolve local $defs."""
    definitions = schema.get("$defs", {})

    def resolve(value: Any, seen: frozenset) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference[len("#/$defs/") :]
                if name in definitions and name not in seen:
                    extra = {
                        key: item
                        for key, item in value.items()
                        if key != "$ref"
                    }
                    return {
                        **resolve(definitions[name], seen | {name}),
                        **resolve(extra, seen),
                    }
            return {
                key: resolve(item, seen)
                for key, item in value.items()
                if key != "$defs"
            }
        if isinstance(value, list):
            return [resolve(item, seen) for item in value]
        return value

    return resolve(schema, frozenset())


def _strip_fence(content: str) -> str:
    """The JSON object of a response, without Markdown or a preamble.

    Some providers wrap the requested JSON in a code fence or add a sentence
    before it.
    """
    text = content.strip()
    fence = "`" * 3
    if text.startswith(fence) and text.endswith(fence) and "\n" in text:
        return text[text.index("\n") + 1 : -3].strip()
    if text.startswith("{"):
        return text
    block = re.search(
        fence + r"(?:json)?[ \t]*\n(.*?)\n?" + fence, text, re.S | re.I
    )
    if block:
        return block.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        return text[start : end + 1]
    return text


def _statuses(value: Any, name: str) -> frozenset:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise LLMError(
            "configuration", f"{name} must be a list of HTTP status codes"
        )
    return frozenset(value)


def _names(value: Any, name: str) -> list:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise LLMError(
            "configuration", f"{name} must be a list of model names"
        )
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class JsonLLM:
    """OpenAI-compatible JSON client with an ordered model ladder.

    Structural validation happens here. Document evidence and review
    semantics remain the parser's responsibility. Inject a sync httpx
    transport for tests.

    The ladder lists models from strongest to weakest. A model whose token
    package is exhausted (HTTP 402, or a known /balance below the reserve) or
    that the account cannot use is retired for the lifetime of this client and
    the same request goes to the next model. Other failures are never retried
    here; the coordinator owns retries.
    """

    demo = False

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        transport: Optional[httpx.BaseTransport] = None,
        stage_models: Optional[Mapping[str, str]] = None,
        config: Optional[Mapping[str, Any]] = None,
        provider: Optional[str] = None,
    ):
        catalog = dict(load_catalog("llm") if config is None else config)
        self.provider = (
            (
                provider
                or os.getenv("LLM_PROVIDER")
                or catalog.get("provider")
                or "openai_compatible"
            )
            .strip()
            .lower()
        )
        if self.provider not in ("openai_compatible", "gigachat"):
            raise LLMError(
                "configuration",
                "LLM_PROVIDER must be openai_compatible or gigachat",
            )
        gigachat = self.provider == "gigachat"
        profile = catalog.get("gigachat", {}) if gigachat else {}
        if not isinstance(profile, dict):
            raise LLMError(
                "configuration", "gigachat profile must be an object"
            )
        self.config = {**catalog, **profile}
        self.model = (
            model if model is not None else os.getenv("LLM_MODEL", "")
        ).strip()
        overrides = dict(stage_models or {})
        if any(stage not in STAGES for stage in overrides):
            raise LLMError(
                "configuration",
                "Model overrides must use extract or review stages",
            )
        ladder = _names(
            os.getenv("LLM_MODEL_LADDER")
            or self.config.get("model_ladder", []),
            "model_ladder",
        )
        self.ladders: Dict[str, list] = {}
        # Routes spread calls over models by task when nothing is pinned:
        # an explicit model or ladder (argument, env) always wins.
        pinned_any = bool(
            overrides
            or self.model
            or os.getenv("LLM_MODEL_LADDER")
            or any(os.getenv(f"LLM_{s.upper()}_MODEL") for s in STAGES)
        )
        routes = self.config.get("model_routes") if not pinned_any else None
        self.short_packet_chars = 0
        if routes:
            self._route_ladders(routes, ladder)
        for stage in STAGES:
            if stage in self.ladders:
                continue
            pinned = (
                overrides.get(stage)
                or os.getenv(f"LLM_{stage.upper()}_MODEL")
                or self.model
            ).strip()
            if pinned:
                # A pinned model still degrades to the weaker ladder models
                # below it.
                weaker = (
                    ladder[ladder.index(pinned) + 1 :]
                    if pinned in ladder
                    else []
                )
                self.ladders[stage] = [pinned, *weaker]
            else:
                self.ladders[stage] = list(ladder)
        if not all(self.ladders.values()):
            raise LLMError(
                "configuration",
                "Set LLM_MODEL, LLM_MODEL_LADDER or both "
                "LLM_EXTRACT_MODEL and LLM_REVIEW_MODEL",
            )
        self.model = self.model or self.ladders["extract"][0]
        fallback = self.config.get("model_fallback", {})
        if not isinstance(fallback, dict):
            raise LLMError("configuration", "model_fallback must be an object")
        self.exhausted_statuses = _statuses(
            fallback.get("exhausted_statuses", [402]), "exhausted_statuses"
        )
        self.unavailable_statuses = _statuses(
            fallback.get("unavailable_statuses", []), "unavailable_statuses"
        )
        self.retired: Dict[str, str] = {}
        self.model_events: list[Dict[str, Any]] = []
        endpoint = base_url or (
            os.getenv("GIGACHAT_BASE_URL")
            if gigachat
            else os.getenv("LLM_BASE_URL")
        )
        self.base_url = self._endpoint(
            endpoint or self.config.get("base_url", ""), "LLM_BASE_URL"
        )
        local = (urlsplit(self.base_url).hostname or "").lower() in (
            "localhost",
            "127.0.0.1",
            "::1",
        )
        if gigachat:
            self.auth_url = self._endpoint(
                os.getenv("GIGACHAT_AUTH_URL")
                or self.config.get("auth_url", ""),
                "GIGACHAT_AUTH_URL",
            )
            self.scope = (
                os.getenv("GIGACHAT_SCOPE") or self.config.get("scope") or ""
            ).strip()
            if not self.scope:
                raise LLMError("configuration", "Set GIGACHAT_SCOPE")
            self._api_key = (
                api_key
                if api_key is not None
                else os.getenv("GIGACHAT_CREDENTIALS")
            )
            if not self._api_key:
                raise LLMError(
                    "configuration",
                    "Set GIGACHAT_CREDENTIALS to the GigaChat "
                    "authorization key",
                )
            self.token_margin = _positive(
                self.config.get("token_refresh_margin_seconds", 60),
                "token_refresh_margin_seconds",
            )
        else:
            self._api_key = (
                api_key if api_key is not None else os.getenv("LLM_API_KEY")
            )
            if not local and not self._api_key:
                raise LLMError(
                    "configuration",
                    "Set LLM_API_KEY for a remote LLM endpoint",
                )
        self._token: Optional[str] = None
        self._token_expires = 0.0
        bundle = (
            os.getenv("GIGACHAT_CA_BUNDLE_FILE") if gigachat else None
        ) or os.getenv("LLM_CA_BUNDLE_FILE")
        if bundle and not Path(bundle).expanduser().is_file():
            raise LLMError("configuration", "CA bundle file does not exist")
        self.verify: Any = str(Path(bundle).expanduser()) if bundle else True
        self.response_mode = self.config.get("response_mode")
        if self.response_mode not in ("json_object", "json_schema"):
            raise LLMError(
                "configuration",
                "Only explicit json_object or json_schema response modes "
                "are supported",
            )
        limits = self.config.get("max_output_tokens", {})
        if not isinstance(limits, dict):
            raise LLMError(
                "configuration", "max_output_tokens must be a stage mapping"
            )
        self.max_output_tokens = {}
        for stage in STAGES:
            value = limits.get(stage)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise LLMError(
                    "configuration",
                    "Each stage requires a positive max_output_tokens integer",
                )
            self.max_output_tokens[stage] = value
        if timeout is not None:
            self.timeout = httpx.Timeout(_positive(timeout, "timeout"))
        else:
            settings = self.config.get("timeouts_seconds", {})
            if not isinstance(settings, dict):
                raise LLMError(
                    "configuration", "timeouts_seconds must be an object"
                )
            self.timeout = httpx.Timeout(
                **{
                    name: _positive(settings.get(name), f"{name} timeout")
                    for name in ("connect", "read", "write", "pool")
                }
            )
        self.temperature = self.config.get("temperature", 0)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise LLMError(
                "configuration",
                "temperature must be a finite number from 0 to 2",
            )
        cache = self.config.get("cache", {})
        if not isinstance(cache, dict) or not isinstance(
            cache.get("enabled", False), bool
        ):
            raise LLMError(
                "configuration", "cache must contain a boolean enabled setting"
            )
        self.cache_enabled = cache.get("enabled", False)
        directory = cache.get("directory")
        if self.cache_enabled and (
            not isinstance(directory, str) or not directory.strip()
        ):
            raise LLMError(
                "configuration", "Enabled cache requires a directory"
            )
        self.cache_dir = (
            Path(directory).expanduser()
            if isinstance(directory, str)
            else None
        )
        self.prices = self.config.get("prices_usd_per_million_tokens", {})
        if not isinstance(self.prices, dict):
            raise LLMError(
                "configuration", "Token prices must be a model mapping"
            )
        balance = self.config.get("balance", {}) if gigachat else {}
        if not isinstance(balance, dict):
            raise LLMError("configuration", "balance must be an object")
        self.balance_enabled = balance.get("enabled", False) is True
        self.balance_refresh = (
            _positive(
                balance.get("refresh_seconds", 300), "balance refresh_seconds"
            )
            if self.balance_enabled
            else 0.0
        )
        self.balance_reserve = balance.get("min_tokens", 0)
        if (
            isinstance(self.balance_reserve, bool)
            or not isinstance(self.balance_reserve, int)
            or self.balance_reserve < 0
        ):
            raise LLMError(
                "configuration",
                "balance min_tokens must be a non-negative integer",
            )
        usage_names = balance.get("usage_names", {})
        if not isinstance(usage_names, dict):
            raise LLMError(
                "configuration", "balance usage_names must be a model mapping"
            )
        self.usage_models = {
            name.casefold(): model
            for model, names in usage_names.items()
            for name in _names(names, "balance usage_names")
        }
        self.balance: Dict[str, int] = {}
        self.balance_status = "pending" if self.balance_enabled else "disabled"
        self._balance_checked = -math.inf
        self.transport = transport
        self.calls: deque = deque(maxlen=MAX_RETAINED_CALLS)
        concurrency = os.getenv("LLM_MAX_CONCURRENCY") or self.config.get(
            "max_concurrent_requests", 1
        )
        try:
            self.max_concurrency = int(concurrency)
        except (TypeError, ValueError):
            self.max_concurrency = 0
        if self.max_concurrency < 1:
            raise LLMError(
                "configuration",
                "max_concurrent_requests must be a positive integer",
            )
        # asyncio primitives belong to one event loop; the CLI, the web job
        # loop and embedding threads may each use this client.
        self._loop_state: weakref.WeakKeyDictionary = (
            weakref.WeakKeyDictionary()
        )

    @staticmethod
    def _endpoint(value: str, name: str) -> str:
        value = value.rstrip("/")
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname
            parsed.port  # Validate malformed ports before sending anything.
        except ValueError:
            raise LLMError(
                "configuration", f"{name} must be an HTTP(S) API base URL"
            ) from None
        if parsed.scheme not in ("http", "https") or not hostname:
            raise LLMError(
                "configuration", f"{name} must be an HTTP(S) API base URL"
            )
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise LLMError(
                "configuration",
                f"{name} must not contain credentials, query or fragment",
            )
        if parsed.scheme == "http" and hostname.lower() not in (
            "localhost",
            "127.0.0.1",
            "::1",
        ):
            raise LLMError(
                "configuration",
                "Plain HTTP is allowed only for a loopback LLM endpoint",
            )
        return value

    @classmethod
    def from_environment(cls, **kwargs: Any) -> JsonLLM:
        """Environment values are read at construction; secrets not logged."""
        return cls(**kwargs)

    def _route_ladders(self, routes: Any, ladder: list) -> None:
        """Ladders per route; each ends with every other known model, so a
        route whose models run out of tokens falls through to the rest."""
        if not isinstance(routes, dict):
            raise LLMError("configuration", "model_routes must be an object")
        short = routes.get("short_packet_chars", 0)
        if isinstance(short, bool) or not isinstance(short, int) or short < 0:
            raise LLMError(
                "configuration",
                "model_routes.short_packet_chars must be a non-negative int",
            )
        self.short_packet_chars = short
        for key in (*STAGES, SHORT_EXTRACT_ROUTE):
            if key not in routes:
                continue
            names = _names(routes[key], f"model_routes.{key}")
            if not names:
                raise LLMError(
                    "configuration", f"model_routes.{key} must name a model"
                )
            self.ladders[key] = [
                *names,
                *(name for name in ladder if name not in names),
            ]

    def _route(self, stage: str, payload: Dict[str, Any]) -> str:
        """Ladder key of a call: short extraction packets (abstracts) take
        their own, usually lighter, route."""
        if (
            stage == "extract"
            and SHORT_EXTRACT_ROUTE in self.ladders
            and self.short_packet_chars
            and len(_json(payload)) <= self.short_packet_chars
        ):
            return SHORT_EXTRACT_ROUTE
        return stage

    @property
    def models(self) -> Dict[str, Optional[str]]:
        """Model each stage would use now; changes as models are retired."""
        return {
            stage: next((m for m in ladder if m not in self.retired), None)
            for stage, ladder in self.ladders.items()
        }

    def _loop_local(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        state = self._loop_state.get(loop)
        if state is None:
            state = {"token": asyncio.Lock()}
            self._loop_state[loop] = state
        return state

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self.timeout,
            transport=self.transport,
            follow_redirects=False,
            verify=self.verify,
        )

    def _retire(self, model: str, reason: str, **details: Any) -> None:
        if model not in self.retired:
            logger.warning("Model %s retired: %s %s", model, reason, details)
            self.retired[model] = reason
            self.model_events.append(
                {
                    "model": model,
                    "event": reason,
                    **details,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def _token_valid(self) -> bool:
        return bool(
            self._token
            and monotonic() < self._token_expires - self.token_margin
        )

    async def _access_token(self, client: httpx.AsyncClient) -> Optional[str]:
        if self.provider != "gigachat":
            return self._api_key
        if self._token_valid():
            return self._token
        # Concurrent documents share one token; only one renews it.
        async with self._loop_local()["token"]:
            if self._token_valid():
                return self._token
            return await self._renew_token(client)

    async def _renew_token(self, client: httpx.AsyncClient) -> str:
        try:
            response = await client.post(
                self.auth_url,
                data={"scope": self.scope},
                headers={
                    "Authorization": "Basic " + self._api_key,
                    "RqUID": str(uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )
        except httpx.TimeoutException:
            raise LLMError(
                "timeout",
                "GigaChat authorization exceeded its timeout",
                retryable=True,
            ) from None
        except httpx.RequestError:
            raise LLMError(
                "transport_error",
                "Cannot reach the GigaChat authorization endpoint",
                retryable=True,
            ) from None
        if not response.is_success:
            retryable = (
                response.status_code == 429
                or 500 <= response.status_code <= 599
            )
            raise LLMError(
                "auth_error",
                f"GigaChat authorization returned HTTP {response.status_code}",
                retryable,
            )
        try:
            data = response.json()
        except ValueError:
            data = None
        token = data.get("access_token") if isinstance(data, dict) else None
        expires = data.get("expires_at") if isinstance(data, dict) else None
        if (
            not isinstance(token, str)
            or not token
            or isinstance(expires, bool)
            or not isinstance(expires, (int, float))
        ):
            raise LLMError(
                "auth_error",
                "GigaChat authorization response has no access token",
            )
        # expires_at is a Unix timestamp; Sber returns milliseconds.
        expires_at = expires / 1000 if expires > 1e11 else float(expires)
        self._token = token
        self._token_expires = monotonic() + max(
            0.0, expires_at - datetime.now(timezone.utc).timestamp()
        )
        logger.debug("GigaChat access token renewed")
        return token

    async def _headers(self, client: httpx.AsyncClient) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        token = await self._access_token(client)
        if token:
            headers["Authorization"] = "Bearer " + token
        return headers

    async def _refresh_balance(self, client: httpx.AsyncClient) -> None:
        """Read prepaid token packages; pay-as-you-go accounts answer 403."""
        if (
            not self.balance_enabled
            or monotonic() - self._balance_checked < self.balance_refresh
        ):
            return
        self._balance_checked = monotonic()
        headers = await self._headers(client)
        try:
            response = await client.get(
                self.base_url + "/balance", headers=headers
            )
        except httpx.RequestError as exc:
            logger.warning("Balance endpoint unreachable: %s", exc)
            self.balance_status = "unreachable"
            return
        try:
            data = response.json() if response.is_success else None
        except ValueError:
            data = None
        entries = data.get("balance") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            # Without packages the 402 answer remains the exhaustion signal.
            self.balance_status = (
                "invalid"
                if response.is_success
                else f"http_{response.status_code}"
            )
            if response.status_code in (401, 403, 404):
                self.balance_enabled = False
            logger.debug("Balance check status: %s", self.balance_status)
            return
        self.balance = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(
                entry.get("usage"), str
            ):
                continue
            model, value = (
                self.usage_models.get(entry["usage"].casefold()),
                entry.get("value"),
            )
            if (
                model
                and not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                self.balance[model] = int(value)
        self.balance_status = "ok"
        logger.debug("Token balance: %s", self.balance)

    async def _select(
        self, stage: str, client: httpx.AsyncClient, route: str = ""
    ) -> str:
        route = route or stage
        if self.balance_enabled:
            await self._refresh_balance(client)
            for model in self.ladders[route]:
                if (
                    model in self.balance
                    and self.balance[model]
                    < self.balance_reserve + self.max_output_tokens[stage]
                ):
                    self._retire(
                        model, "low_balance", balance=self.balance[model]
                    )
        model = self.models[route]
        if model is None:
            raise LLMError(
                "models_exhausted",
                "Every model in the ladder is exhausted or unavailable: "
                + ", ".join(
                    f"{name}={self.retired[name]}"
                    for name in self.ladders[route]
                ),
            )
        return model

    def _cost(self, model: str, tokens: Dict[str, int]) -> Optional[float]:
        prices = self.prices.get(model)
        if not isinstance(prices, dict) or not {
            "prompt_tokens",
            "completion_tokens",
        }.issubset(tokens):
            return None
        rates = [prices.get("input"), prices.get("output")]
        if any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate < 0
            for rate in rates
        ):
            return None
        prompt = tokens["prompt_tokens"]
        cached = tokens.get("cached_tokens", 0)
        if cached > prompt:
            return None
        cached_rate = prices.get("cached_input")
        if cached:
            if (
                isinstance(cached_rate, bool)
                or not isinstance(cached_rate, (int, float))
                or not math.isfinite(cached_rate)
                or cached_rate < 0
            ):
                return None
        return (
            (prompt - cached) * rates[0]
            + cached * (cached_rate or 0)
            + tokens["completion_tokens"] * rates[1]
        ) / 1_000_000

    async def _open(self, stack: AsyncExitStack) -> httpx.AsyncClient:
        # The limit bounds concurrent requests to the provider (GigaChat
        # personal accounts allow a single stream) across the process.
        await stack.enter_async_context(
            _RequestSlot(
                _process_limit(
                    self.provider, self.base_url, self.max_concurrency
                )
            )
        )
        return await stack.enter_async_context(self._client())

    async def generate(
        self,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        *,
        stage: str = "extract",
    ) -> T:
        _stage(stage)
        route = self._route(stage, payload)
        async with AsyncExitStack() as stack:
            client = await self._open(stack)
            while True:
                try:
                    model = await self._select(stage, client, route)
                    return await self._attempt(
                        client, schema, system, payload, stage, model, route
                    )
                except LLMError as exc:
                    if exc.code not in (
                        "model_exhausted",
                        "model_unavailable",
                    ):
                        raise
                    # Retired for every route: its token package is shared.
                    self._retire(model, exc.code, stage=stage, route=route)

    async def _attempt(
        self,
        client: httpx.AsyncClient,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        stage: str,
        model: str,
        route: str = "",
    ) -> T:
        route = route or stage
        schema_json = schema.model_json_schema()
        strict = self.config.get("json_schema_strict") is True
        if self.provider == "gigachat" and strict:
            # GigaChat treats a schema without root-level required fields as
            # an unconstrained JSON object, even when strict mode is enabled.
            schema_json.setdefault(
                "required", list(schema_json.get("properties", {}))
            )
        system_message = (
            system + "\n\nReturn exactly one JSON object. No Markdown. "
            "The JSON object must match this schema:\n" + _json(schema_json)
        )
        body: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": _json(payload)},
            ],
            "max_tokens": self.max_output_tokens[stage],
            "temperature": self.temperature,
        }
        if self.response_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "schema": _inline_refs(schema_json),
                "strict": self.config.get("json_schema_strict") is True,
            }
        else:
            body["response_format"] = {"type": "json_object"}
        request_hash = _hash(
            {
                "endpoint": self.base_url,
                "stage": stage,
                "schema": schema_json,
                "request": body,
            }
        )
        started = perf_counter()
        call: Dict[str, Any] = {
            "provider": "gigachat"
            if self.provider == "gigachat"
            else "json_llm",
            "stage": stage,
            "route": route,
            "model": model,
            "ladder_position": self.ladders[route].index(model),
            "schema": schema.__name__,
            "prompt_sha256": _hash(body["messages"]),
            "request_sha256": request_hash,
            "cache_hit": False,
            "tokens": {},
            "estimated_cost_usd": None,
            "status": "started",
        }
        _record(self.calls, call)
        cache_path = (
            self.cache_dir / (request_hash + ".json")
            if self.cache_enabled and self.cache_dir is not None
            else None
        )
        try:
            if cache_path is not None and cache_path.is_file():
                call["cache_hit"] = True
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    raise LLMError(
                        "invalid_cache",
                        "Cached LLM response cannot be read as JSON",
                    ) from None
                if (
                    not isinstance(cached, dict)
                    or cached.get("request_sha256") != request_hash
                    or "response" not in cached
                ):
                    raise LLMError(
                        "invalid_cache",
                        "Cached LLM response does not match the request",
                    )
                result = _validate(schema, cached["response"])
                call.update(
                    status="ok",
                    cached_response_tokens=_tokens(cached.get("usage")),
                )
                # This call did not request tokens. Historical response
                # usage is separate; do not attribute its cost to this run.
                return result
            response = await client.post(
                self.base_url + "/chat/completions",
                headers=await self._headers(client),
                json=body,
            )
            if response.status_code == 401 and self.provider == "gigachat":
                # The access token lives 30 minutes; renew it once and resend.
                logger.debug("GigaChat token rejected, renewing it once")
                self._token = None
                response = await client.post(
                    self.base_url + "/chat/completions",
                    headers=await self._headers(client),
                    json=body,
                )
            call["http_status"] = response.status_code
            if response.status_code in self.exhausted_statuses:
                raise LLMError(
                    "model_exhausted",
                    f"Model has no tokens left (HTTP {response.status_code})",
                )
            if response.status_code in self.unavailable_statuses:
                raise LLMError(
                    "model_unavailable",
                    "Model is unavailable to this account "
                    f"(HTTP {response.status_code})",
                )
            if not response.is_success:
                retryable = (
                    response.status_code in (408, 429)
                    or 500 <= response.status_code <= 599
                )
                raise LLMError(
                    "http_error",
                    f"LLM endpoint returned HTTP {response.status_code}",
                    retryable,
                    _retry_after(response.headers.get("Retry-After"))
                    if retryable
                    else None,
                )
            try:
                data = response.json()
            except ValueError:
                raise LLMError(
                    "invalid_response",
                    "LLM endpoint returned non-JSON response",
                ) from None
            if not isinstance(data, dict):
                raise LLMError(
                    "invalid_response", "LLM response must be an object"
                )
            call["tokens"] = _tokens(data.get("usage"))
            call["estimated_cost_usd"] = self._cost(model, call["tokens"])
            if model in self.balance and "total_tokens" in call["tokens"]:
                self.balance[model] = max(
                    0, self.balance[model] - call["tokens"]["total_tokens"]
                )
            if data.get("error"):
                raise LLMError(
                    "provider_error", "LLM endpoint reported an error"
                )
            choices = data.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
            ):
                raise LLMError(
                    "invalid_response", "LLM response has no choice"
                )
            choice = choices[0]
            call["finish_reason"] = choice.get("finish_reason")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise LLMError(
                    "invalid_response", "LLM response has no message"
                )
            content = message.get("content")
            if isinstance(content, str):
                call["response_chars"] = len(content)
            # GigaChat reports its content filter as finish_reason=blacklist.
            if message.get("refusal") or choice.get("finish_reason") in (
                "content_filter",
                "blacklist",
            ):
                raise LLMError("refusal", "LLM declined the parser request")
            if choice.get("finish_reason") != "stop":
                raise LLMError(
                    "incomplete_response",
                    "LLM did not finish a complete JSON response",
                )
            if not isinstance(content, str) or not content.strip():
                raise LLMError(
                    "invalid_response", "LLM response contains no text"
                )
            result = _validate(schema, _strip_fence(content))
            call["status"] = "ok"
            if cache_path is not None:
                self._write_cache(
                    cache_path,
                    {
                        "request_sha256": request_hash,
                        "response": result.model_dump(mode="json"),
                        "usage": call["tokens"],
                    },
                    call,
                )
            return result
        except httpx.TimeoutException:
            call.update(status="error", error_code="timeout")
            raise LLMError(
                "timeout", "LLM request exceeded its timeout", retryable=True
            ) from None
        except httpx.RequestError:
            call.update(status="error", error_code="transport_error")
            raise LLMError(
                "transport_error",
                "Cannot reach the LLM endpoint",
                retryable=True,
            ) from None
        except LLMError as exc:
            call.update(
                status="error", error_code=exc.code, retryable=exc.retryable
            )
            raise
        finally:
            call["duration_ms"] = round((perf_counter() - started) * 1000)
            _log_call(call)

    async def embed(
        self, texts: Sequence[str], model: str
    ) -> list[list[float]]:
        """Vectors from the provider's /embeddings endpoint, in input order.

        GigaChat serves embeddings with the same OAuth token as chat; the
        model ladder does not apply to embedding models.
        """
        texts = list(texts)
        if not texts:
            return []
        started = perf_counter()
        call: Dict[str, Any] = {
            "provider": "gigachat"
            if self.provider == "gigachat"
            else "json_llm",
            "stage": "embed",
            "model": model,
            "inputs": len(texts),
            "cache_hit": False,
            "tokens": {},
            "estimated_cost_usd": None,
            "status": "started",
        }
        _record(self.calls, call)
        body = {"model": model, "input": texts}
        try:
            async with AsyncExitStack() as stack:
                client = await self._open(stack)
                response = await client.post(
                    self.base_url + "/embeddings",
                    headers=await self._headers(client),
                    json=body,
                )
                if response.status_code == 401 and self.provider == "gigachat":
                    self._token = None
                    response = await client.post(
                        self.base_url + "/embeddings",
                        headers=await self._headers(client),
                        json=body,
                    )
            call["http_status"] = response.status_code
            if not response.is_success:
                retryable = (
                    response.status_code in (408, 429)
                    or 500 <= response.status_code <= 599
                )
                raise LLMError(
                    "http_error",
                    f"Embedding endpoint returned HTTP {response.status_code}",
                    retryable,
                    _retry_after(response.headers.get("Retry-After"))
                    if retryable
                    else None,
                )
            try:
                data = response.json()
            except ValueError:
                data = None
            items = data.get("data") if isinstance(data, dict) else None
            if not isinstance(items, list) or len(items) != len(texts):
                raise LLMError(
                    "invalid_response",
                    "Embedding response does not match the request",
                )
            vectors: list[Optional[list[float]]] = [None] * len(texts)
            total = 0
            for position, item in enumerate(items):
                index = (
                    item.get("index", position)
                    if isinstance(item, dict)
                    else None
                )
                vector = (
                    item.get("embedding") if isinstance(item, dict) else None
                )
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(texts)
                    or vectors[index] is not None
                    or not isinstance(vector, list)
                    or not vector
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        for value in vector
                    )
                ):
                    raise LLMError(
                        "invalid_response",
                        "Embedding response contains an invalid vector",
                    )
                vectors[index] = [float(value) for value in vector]
                total += _tokens(item.get("usage")).get("prompt_tokens", 0)
            if total:
                call["tokens"] = {"prompt_tokens": total}
            call["status"] = "ok"
            return vectors  # type: ignore[return-value]
        except httpx.TimeoutException:
            call.update(status="error", error_code="timeout")
            raise LLMError(
                "timeout",
                "Embedding request exceeded its timeout",
                retryable=True,
            ) from None
        except httpx.RequestError:
            call.update(status="error", error_code="transport_error")
            raise LLMError(
                "transport_error",
                "Cannot reach the embedding endpoint",
                retryable=True,
            ) from None
        except LLMError as exc:
            call.update(
                status="error", error_code=exc.code, retryable=exc.retryable
            )
            raise
        finally:
            call["duration_ms"] = round((perf_counter() - started) * 1000)
            _log_call(call)

    @staticmethod
    def _write_cache(
        path: Path, value: Dict[str, Any], call: Dict[str, Any]
    ) -> None:
        temporary = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(_json(value))
            temporary.replace(path)
        except OSError as exc:
            # A cache storage failure must not discard a validated paid answer.
            logger.warning("Cannot write LLM cache %s: %s", path, exc)
            call["cache_write_failed"] = True
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    call["cache_write_failed"] = True


class ReplayProvider:
    """Play supplied fixtures in sequence; never perform inference."""

    demo = True

    def __init__(self, answers: Any, *, label: str = "recorded-fixture"):
        if isinstance(answers, (str, Path)):
            try:
                loaded = json.loads(
                    Path(answers).read_text(encoding="utf-8-sig")
                )
            except (OSError, ValueError):
                raise LLMError(
                    "configuration", "Cannot read replay fixture JSON"
                ) from None
            answers = (
                loaded.get("answers") if isinstance(loaded, dict) else loaded
            )
        if not isinstance(answers, Sequence) or isinstance(
            answers, (str, bytes)
        ):
            raise LLMError(
                "configuration", "Replay fixture must contain an answers list"
            )
        self.answers = list(answers)
        self.label = label
        self.position = 0
        self.calls: deque = deque(maxlen=MAX_RETAINED_CALLS)

    @classmethod
    def from_file(cls, path: Any) -> ReplayProvider:
        return cls(path, label=Path(path).name)

    async def generate(
        self,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        *,
        stage: str = "extract",
    ) -> T:
        _stage(stage)
        started = perf_counter()
        call: Dict[str, Any] = {
            "provider": "replay",
            "stage": stage,
            "model": None,
            "schema": schema.__name__,
            "answer_index": self.position,
            "demo": True,
            "cache_hit": False,
            "tokens": {},
            "estimated_cost_usd": None,
            "prompt_sha256": _hash(system),
            "request_sha256": _hash(
                {
                    "stage": stage,
                    "schema": schema.model_json_schema(),
                    "system": system,
                    "payload": payload,
                }
            ),
            "status": "started",
        }
        _record(self.calls, call)
        try:
            if self.position >= len(self.answers):
                raise LLMError(
                    "replay_exhausted", "No recorded answer remains"
                )
            answer = self.answers[self.position]
            self.position += 1
            if (
                isinstance(answer, dict)
                and "response" in answer
                and ("stage" in answer or "schema" in answer)
            ):
                if "stage" in answer and answer["stage"] != stage:
                    raise LLMError(
                        "replay_stage_mismatch",
                        "Recorded answer belongs to a different parser stage",
                    )
                if "schema" in answer and answer["schema"] != schema.__name__:
                    raise LLMError(
                        "replay_schema_mismatch",
                        "Recorded answer belongs to a different schema",
                    )
                answer = answer["response"]
            result = _validate(schema, answer)
            call["status"] = "ok"
            return result
        except LLMError as exc:
            call.update(
                status="error", error_code=exc.code, retryable=exc.retryable
            )
            raise
        finally:
            call["duration_ms"] = round((perf_counter() - started) * 1000)
            _log_call(call)
