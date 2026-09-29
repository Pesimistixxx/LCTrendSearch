"""One synchronous, schema-validated LLM request or explicit offline replay.

The coordinator owns retries, context tools and total call budgets. This
module never retries a failed answer, follows redirects or substitutes
fabricated answers. It only moves a request down the configured model ladder
when a model has no tokens left or is not available to the account.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import re
import threading
import weakref
from collections import deque
from collections.abc import Mapping, Sequence
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
from .stats import STATS

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


# One request limit per provider endpoint and key for the whole process: the
# CLI, the web job loop, the resolver's own client and embedding threads all
# count against LLM_MAX_CONCURRENCY together (B-10). Each key of a KeyPool
# has its own limit: providers count concurrent streams per account. The
# limit belongs to the key, not to the client: the first client of a key
# fixes it, and a later client asking for more (a single-key JsonLLM on the
# same credentials, another pool) shares it instead of opening a second
# gate onto the same account.
_PROCESS_LIMITS: Dict[
    Tuple[str, str, str], Tuple[int, threading.BoundedSemaphore]
] = {}
_PROCESS_LIMITS_LOCK = threading.Lock()


# Key -> monotonic time it may be used again after HTTP 429, and its last
# pause. Process-wide like the limits: every pool built from one keys file
# (jobs, the semantic layer, status checks) must leave a resting key alone.
_KEY_RESTING: Dict[Tuple[str, str, str], float] = {}
_KEY_PAUSE: Dict[Tuple[str, str, str], float] = {}


# Which keys embed, which have no embeddings package: known once per
# process, because the semantic layer and the extraction build separate
# pools from one keys file. Keyed like the limits, by the key's hash.
_EMBEDDING_KEYS: Dict[Tuple[str, str, str], bool] = {}


def _embedding_key(member: Any) -> Tuple[str, str, str]:
    return (member.provider, member.base_url, member._key_id)


def reset_key_knowledge() -> None:
    """Forget which keys embed, rest and their limits (tests, or after new
    packages are bought). A request holding a slot releases the semaphore
    it acquired, so forgetting a limit never breaks a running request."""
    with _PROCESS_LIMITS_LOCK:
        _EMBEDDING_KEYS.clear()
        _KEY_RESTING.clear()
        _KEY_PAUSE.clear()
        _PROCESS_LIMITS.clear()


def _process_limit(
    provider: str, base_url: str, key_id: str, limit: int
) -> threading.BoundedSemaphore:
    with _PROCESS_LIMITS_LOCK:
        key = (provider, base_url, key_id)
        if key not in _PROCESS_LIMITS:
            _PROCESS_LIMITS[key] = (limit, threading.BoundedSemaphore(limit))
        fixed, semaphore = _PROCESS_LIMITS[key]
    if fixed != limit:
        _log_once(
            logging.WARNING,
            "LLM key %s: a client asked for %d concurrent requests, the key "
            "keeps its limit of %d",
            key_id[:8] or "default",
            limit,
            fixed,
        )
    return semaphore


async def _take_slot(semaphore: threading.BoundedSemaphore) -> int:
    """Hold one process-wide request slot; returns the wait in ms.

    A thread semaphore works across loops and threads; polling it without
    blocking the event loop keeps a cancelled waiter from ever owning a
    slot it cannot release.
    """
    queued = perf_counter()
    STATS.wait(1)
    try:
        while not semaphore.acquire(blocking=False):
            await asyncio.sleep(0.01)
    finally:
        STATS.wait(-1)
    return round((perf_counter() - queued) * 1000)


def document_workers(requested: int, provider: Any) -> int:
    """Documents processed at once: never more than the LLM provider runs
    requests at once (one per key of a key pool, by its ``workers``), so
    each worker has its own key and no document queues behind another on
    a busy account. Without a provider (no model calls) nothing caps it."""
    capacity = getattr(provider, "max_concurrency", None)
    if isinstance(capacity, bool) or not isinstance(capacity, int):
        return requested
    return max(1, min(requested, capacity))


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
    """Log the call summary only: prompts and answers may hold source text.

    Chat calls are logged at INFO with their timing (queue: waiting for a
    free key; request: the provider's answer); embeddings, fast and
    frequent, at DEBUG unless they are slow.
    """
    tokens = call.get("tokens") or {}
    duration = (call.get("duration_ms") or 0) / 1000
    output = tokens.get("completion_tokens") or 0
    slow_embedding = call.get("stage") == "embed" and duration >= 5
    level = (
        logging.INFO
        if call.get("stage") != "embed" or slow_embedding
        else logging.DEBUG
    )
    logger.log(
        level,
        "LLM %s %s key=%s model=%s queue=%.1fs request=%.1fs "
        "in=%s (cached %s) out=%s%s%s",
        call.get("stage"),
        call.get("status")
        + (f"/{call['error_code']}" if call.get("error_code") else ""),
        call.get("key") or "default",
        call.get("model"),
        (call.get("queue_ms") or 0) / 1000,
        duration,
        tokens.get("prompt_tokens", "-"),
        tokens.get("precached_prompt_tokens", 0),
        output or "-",
        f" ({output / duration:.0f} tok/s)" if output and duration else "",
        " cache_hit" if call.get("cache_hit") else "",
    )
    STATS.record(call)


class LLMError(RuntimeError):
    """Safe machine-readable failure without raw HTTP bodies or secrets."""

    def __init__(
        self,
        code: str,
        message: str,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        status: Optional[int] = None,
    ):
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        # The HTTP status of the provider's answer, when there was one.
        self.status = status
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
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        # GigaChat: prompt tokens served from its cache (X-Session-ID),
        # not billed.
        "precached_prompt_tokens",
    ):
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
        scope: Optional[str] = None,
        max_concurrency: Optional[int] = None,
        key_name: Optional[str] = None,
    ):
        catalog = dict(load_catalog("llm") if config is None else config)
        # Names this key in call audits and model events of a KeyPool.
        self.key_name = key_name
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
                scope
                or os.getenv("GIGACHAT_SCOPE")
                or self.config.get("scope")
                or ""
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
        self.schema_free_models = frozenset(
            _names(
                self.config.get("schema_free_models", []),
                "schema_free_models",
            )
        )
        # Of those, the models whose constrained decoding is sound: a broken
        # unconstrained answer is resent once under the strict schema.
        self.schema_fallback_models = frozenset(
            _names(
                self.config.get("schema_fallback_models", []),
                "schema_fallback_models",
            )
        )
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
        concurrency = (
            max_concurrency
            or os.getenv("LLM_MAX_CONCURRENCY")
            or self.config.get("max_concurrent_requests", 1)
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
        STATS.register(self.key_name or "default", self.max_concurrency)
        # Keys share nothing: a hash, not the secret, tells their limits apart.
        self._key_id = (
            sha256(self._api_key.encode("utf-8")).hexdigest()[:16]
            if self._api_key
            else ""
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
    def from_environment(cls, **kwargs: Any) -> Any:
        """Environment values are read at construction; secrets not logged.

        With GIGACHAT_KEYS_FILE set, a GigaChat provider is a KeyPool of
        every key in that file instead of the single GIGACHAT_CREDENTIALS.
        """
        path = os.getenv("GIGACHAT_KEYS_FILE", "").strip()
        provider = (
            kwargs.get("provider")
            or os.getenv("LLM_PROVIDER")
            or load_catalog("llm").get("provider")
            or ""
        ).strip().lower()
        if path and provider == "gigachat" and "api_key" not in kwargs:
            return KeyPool.from_file(path, **kwargs)
        # Why no pool: a single key serves every request of this provider.
        if not path:
            reason = "GIGACHAT_KEYS_FILE is not set"
        elif provider != "gigachat":
            reason = f"LLM_PROVIDER is {provider or 'empty'}, not gigachat"
        else:
            reason = "an explicit api_key was passed"
        _log_once(
            logging.INFO, "LLM single key, no key pool: %s", reason
        )
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
                    **({"key": self.key_name} if self.key_name else {}),
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
                status=response.status_code,
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

    def _limit(self) -> threading.BoundedSemaphore:
        # The limit bounds concurrent requests to the provider (GigaChat
        # personal accounts allow a single stream) across the process.
        return _process_limit(
            self.provider, self.base_url, self._key_id, self.max_concurrency
        )

    def available(self, route: str) -> bool:
        """Whether a model of this route still has tokens on this key."""
        return any(m not in self.retired for m in self.ladders[route])

    async def generate(
        self,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        *,
        stage: str = "extract",
    ) -> T:
        _stage(stage)
        slot = self._limit()
        queue_ms = await _take_slot(slot)
        try:
            return await self._generate(
                schema, system, payload, stage, queue_ms
            )
        finally:
            slot.release()

    async def _generate(
        self,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        stage: str,
        queue_ms: int = 0,
    ) -> T:
        """One request; the caller holds this key's request slot."""
        route = self._route(stage, payload)
        key = self.key_name or "default"
        STATS.begin(key)
        try:
            async with self._client() as client:
                return await self._ladder(
                    client, schema, system, payload, stage, route, queue_ms
                )
        finally:
            STATS.end(key)

    async def _ladder(
        self,
        client: httpx.AsyncClient,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        stage: str,
        route: str,
        queue_ms: int,
    ) -> T:
        """The request down the model ladder while models run out."""
        while True:
            try:
                model = await self._select(stage, client, route)
                try:
                    return await self._attempt(
                        client,
                        schema,
                        system,
                        payload,
                        stage,
                        model,
                        route,
                        queue_ms,
                    )
                except LLMError as exc:
                    if (
                        exc.code != "invalid_schema"
                        or model not in self.schema_fallback_models
                    ):
                        raise
                    logger.info(
                        "LLM %s: unconstrained %s answer is not valid %s; "
                        "resending under the schema",
                        stage,
                        model,
                        schema.__name__,
                    )
                    return await self._attempt(
                        client,
                        schema,
                        system,
                        payload,
                        stage,
                        model,
                        route,
                        0,
                        constrained=True,
                    )
            except LLMError as exc:
                if exc.code not in ("model_exhausted", "model_unavailable"):
                    raise
                # Retired for every route: its token package is shared.
                self._retire(model, exc.code, stage=stage, route=route)
                # The wait for a slot belongs to the first attempt only.
                queue_ms = 0

    async def _attempt(
        self,
        client: httpx.AsyncClient,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        stage: str,
        model: str,
        route: str = "",
        queue_ms: int = 0,
        constrained: bool = False,
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
            "Write compact JSON on one line, without indentation. "
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
        if model in self.schema_free_models and not constrained:
            # Constrained decoding of these models fills JSON indentation
            # with stray words on long answers, or (3-Ultra) indents it:
            # 2-5x the output tokens of the compact answer the prompt asks
            # for. The schema in the system message and validation keep the
            # shape.
            pass
        elif self.response_mode == "json_schema":
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
            **({"key": self.key_name} if self.key_name else {}),
            "ladder_position": self.ladders[route].index(model),
            "schema": schema.__name__,
            **({"constrained": True} if constrained else {}),
            "prompt_sha256": _hash(body["messages"]),
            "request_sha256": request_hash,
            "cache_hit": False,
            "tokens": {},
            "estimated_cost_usd": None,
            "queue_ms": queue_ms,
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
            # GigaChat caches a session's prompt prefix: calls sharing the
            # model and the long system message (prompt + schema) reuse it,
            # and precached tokens are not billed.
            session = (
                {"X-Session-ID": _hash([model, system_message])[:32]}
                if self.provider == "gigachat"
                else {}
            )
            response = await client.post(
                self.base_url + "/chat/completions",
                headers={**await self._headers(client), **session},
                json=body,
            )
            if response.status_code == 401 and self.provider == "gigachat":
                # The access token lives 30 minutes; renew it once and resend.
                logger.debug("GigaChat token rejected, renewing it once")
                self._token = None
                response = await client.post(
                    self.base_url + "/chat/completions",
                    headers={**await self._headers(client), **session},
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
                    status=response.status_code,
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
        slot = self._limit()
        queue_ms = await _take_slot(slot)
        try:
            return await self._embed(texts, model, queue_ms)
        finally:
            slot.release()

    async def _embed(
        self, texts: list, model: str, queue_ms: int = 0
    ) -> list[list[float]]:
        """One embedding request; the caller holds this key's slot."""
        key = self.key_name or "default"
        STATS.begin(key)
        try:
            return await self._embed_request(texts, model, queue_ms)
        finally:
            STATS.end(key)

    async def _embed_request(
        self, texts: list, model: str, queue_ms: int
    ) -> list[list[float]]:
        started = perf_counter()
        call: Dict[str, Any] = {
            "provider": "gigachat"
            if self.provider == "gigachat"
            else "json_llm",
            "stage": "embed",
            "model": model,
            **({"key": self.key_name} if self.key_name else {}),
            "inputs": len(texts),
            "cache_hit": False,
            "tokens": {},
            "estimated_cost_usd": None,
            "queue_ms": queue_ms,
            "status": "started",
        }
        _record(self.calls, call)
        body = {"model": model, "input": texts}
        try:
            async with self._client() as client:
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
                    status=response.status_code,
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


_LOGGED_ONCE: set = set()
_LOGGED_ONCE_LOCK = threading.Lock()


def _log_once(level: int, message: str, *args: Any) -> None:
    """Log a configuration line once per distinct text: providers are built
    per request, so a repeated line would flood the log."""
    text = message % args
    with _LOGGED_ONCE_LOCK:
        if text in _LOGGED_ONCE:
            return
        _LOGGED_ONCE.add(text)
    logger.log(level, "%s", text)


def _client_id(credentials: str) -> str:
    """client_id of a GigaChat authorization key (base64 of id:secret)."""
    try:
        decoded = base64.b64decode(credentials, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""
    client_id, separator, _ = decoded.partition(":")
    return client_id.strip() if separator else ""


def load_keys(path: Any) -> list[Dict[str, Any]]:
    """Enabled keys of a GIGACHAT_KEYS_FILE; errors never quote secrets.

    Fields follow the GigaChat API project settings: ``client_id``,
    ``auth_key`` (the Authorization key, base64 of
    ``client_id:client_secret``; ``credentials`` is an alias) or
    ``client_secret`` instead of it, and ``scope`` (overrides
    GIGACHAT_SCOPE). ``workers`` is how many requests the key may run at
    once.
    """
    try:
        data = json.loads(
            Path(path).expanduser().read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        raise LLMError(
            "configuration", "Cannot read GIGACHAT_KEYS_FILE as JSON"
        ) from None
    entries = data.get("keys") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise LLMError(
            "configuration", "GIGACHAT_KEYS_FILE must contain a keys list"
        )
    keys: list[Dict[str, Any]] = []
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise LLMError(
                "configuration", f"Key {position} must be an object"
            )
        if entry.get("enabled", True) is False:
            _log_once(
                logging.WARNING,
                "LLM key pool: key %d (%s) skipped: \"enabled\": false in %s",
                position,
                entry.get("name") or "unnamed",
                Path(path).expanduser().resolve(),
            )
            continue
        client_id = str(entry.get("client_id") or "").strip()
        secret = str(entry.get("client_secret") or "").strip()
        credentials = str(
            entry.get("auth_key") or entry.get("credentials") or ""
        ).strip()
        if not credentials and _client_id(secret):
            # An Authorization key pasted into client_secret.
            credentials = secret
        elif not credentials and client_id and secret:
            credentials = base64.b64encode(
                f"{client_id}:{secret}".encode("utf-8")
            ).decode("ascii")
        if not credentials:
            raise LLMError(
                "configuration",
                f"Key {position} needs auth_key, or client_id "
                "and client_secret",
            )
        embedded = _client_id(credentials)
        if not embedded:
            raise LLMError(
                "configuration",
                f"Key {position} auth_key is not a GigaChat Authorization key",
            )
        if client_id and embedded != client_id:
            raise LLMError(
                "configuration",
                f"Key {position} auth_key belongs to another client_id",
            )
        workers = entry.get("workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or (
            workers < 1
        ):
            raise LLMError(
                "configuration",
                f"Key {position} workers must be a positive integer",
            )
        scope = entry.get("scope")
        if scope is not None and not isinstance(scope, str):
            raise LLMError(
                "configuration", f"Key {position} scope must be a string"
            )
        name = str(entry.get("name") or embedded[:8] or f"key-{position}")
        if any(key["credentials"] == credentials for key in keys):
            raise LLMError(
                "configuration", f"Key {position} ({name}) is listed twice"
            )
        keys.append(
            {
                "name": name,
                "credentials": credentials,
                "workers": workers,
                "scope": scope or None,
            }
        )
    if not keys:
        raise LLMError(
            "configuration", "GIGACHAT_KEYS_FILE has no enabled keys"
        )
    return keys


# A rate-limited key rests this long when the answer names no
# Retry-After, doubling while it keeps answering 429, up to the maximum.
KEY_COOLDOWN_SECONDS = 5.0
KEY_COOLDOWN_MAX_SECONDS = 60.0


def rate_limited(exc: Exception) -> bool:
    """The provider refused the request (HTTP 429): no model work done."""
    return getattr(exc, "status", None) == 429


def rejected_key(exc: Exception) -> bool:
    """The key itself is refused: HTTP 401 after the one token renewal,
    or an authorization failure that no retry fixes. A 403 is not: for
    GigaChat it means a model outside the key's plan (llm.json
    gigachat.model_fallback), which retires that model only."""
    return getattr(exc, "status", None) == 401 or (
        getattr(exc, "code", None) == "auth_error"
        and not getattr(exc, "retryable", False)
    )


class KeyPool:
    """Several provider keys behind one provider.

    Each key is a JsonLLM with its own OAuth token, token balance, retired
    models and request limit (``workers``: requests the account runs at
    once). A request takes the first free slot of a key whose route still
    has a model with tokens, so N keys serve N times as many requests at
    once. A key whose models run out passes the request to the other keys.
    A key answering HTTP 429 rests (Retry-After, else a doubling pause) and
    the request moves to another key at once, so one rate-limited account
    neither fails requests nor spends the caller's retries.

    Embeddings are a separate package: a key without it (402, or the model
    outside its plan) is remembered and skipped for embeddings. Once the
    pool knows which keys embed, chat leaves them to embeddings while
    another key can chat, so the semantic layer does not queue behind
    extraction on the only key that embeds.
    """

    demo = False

    def __init__(self, members: Sequence[JsonLLM]):
        self.members = list(members)
        if not self.members:
            raise LLMError("configuration", "KeyPool needs at least one key")
        first = self.members[0]
        self.provider, self.base_url = first.provider, first.base_url
        self.ladders = first.ladders
        self.short_packet_chars = first.short_packet_chars
        self.max_concurrency = sum(m.max_concurrency for m in self.members)
        # One audit for the pool: documents slice these by offset.
        self.calls: deque = deque(maxlen=MAX_RETAINED_CALLS)
        self.model_events: list[Dict[str, Any]] = []
        for member in self.members:
            member.calls, member.model_events = self.calls, self.model_events
        self._next = 0
        # A rate-limited key rests process-wide: _KEY_RESTING, _KEY_PAUSE.
        # Keys refused for the run (revoked or expired credentials).
        self._rejected: set = set()

    @classmethod
    def from_file(cls, path: Any, **kwargs: Any) -> KeyPool:
        kwargs["provider"] = kwargs.get("provider") or "gigachat"
        pool = cls(
            [
                JsonLLM(
                    api_key=key["credentials"],
                    scope=key["scope"],
                    max_concurrency=key["workers"],
                    key_name=key["name"],
                    **kwargs,
                )
                for key in load_keys(path)
            ]
        )
        logger.info(
            "LLM key pool: %d keys, %d concurrent requests",
            len(pool.members),
            pool.max_concurrency,
        )
        # Which keys and limits: client_id prefixes only, never secrets.
        _log_once(
            logging.INFO,
            "LLM key pool from %s: %s",
            Path(path).expanduser().resolve(),
            ", ".join(
                f"{m.key_name}(client {_client_id(m._api_key)[:8]}, "
                f"workers {m.max_concurrency}, scope {m.scope})"
                for m in pool.members
            ),
        )
        if len(pool.members) == 1:
            _log_once(
                logging.WARNING,
                "LLM key pool has a single enabled key: every request goes "
                "through %s; enable more keys in %s",
                pool.members[0].key_name,
                Path(path).expanduser().resolve(),
            )
        return pool

    @property
    def models(self) -> Dict[str, Optional[str]]:
        """Model each route would use now on the first key that has one."""
        return {
            route: next(
                (
                    member.models[route]
                    for member in self.members
                    if member.models[route]
                ),
                None,
            )
            for route in self.ladders
        }

    def _embeds(self, member: JsonLLM) -> Optional[bool]:
        """True: this key embeds; False: it has no embeddings; None: not
        known yet (process-wide, see _EMBEDDING_KEYS)."""
        return _EMBEDDING_KEYS.get(_embedding_key(member))

    def _learn_embeddings(self, member: JsonLLM, embeds: bool) -> None:
        with _PROCESS_LIMITS_LOCK:
            _EMBEDDING_KEYS[_embedding_key(member)] = embeds

    @staticmethod
    def _awake(member: JsonLLM) -> bool:
        return _KEY_RESTING.get(_embedding_key(member), 0) <= monotonic()

    def _rest(self, member: JsonLLM, exc: LLMError) -> None:
        if rejected_key(exc):
            self._rejected.add(self.members.index(member))
            logger.warning(
                "Key %s is refused (%s); not used again in this run",
                member.key_name,
                exc.code,
            )
            return
        key = _embedding_key(member)
        with _PROCESS_LIMITS_LOCK:
            pause = exc.retry_after or min(
                _KEY_PAUSE.get(key, KEY_COOLDOWN_SECONDS / 2) * 2,
                KEY_COOLDOWN_MAX_SECONDS,
            )
            _KEY_PAUSE[key] = pause
            _KEY_RESTING[key] = max(
                _KEY_RESTING.get(key, 0), monotonic() + pause
            )
        logger.info(
            "Key %s is rate limited (HTTP 429); resting %.0fs",
            member.key_name,
            pause,
        )

    @staticmethod
    def _rested(member: JsonLLM) -> None:
        with _PROCESS_LIMITS_LOCK:
            _KEY_PAUSE.pop(_embedding_key(member), None)

    async def _acquire(
        self, usable: Any
    ) -> Optional[Tuple[JsonLLM, threading.BoundedSemaphore, int]]:
        """A free slot of a usable key and the wait for it in ms, or None
        when no key is usable. A rate-limited key waits for its pause to
        end; a refused one is skipped, and a pool of refused keys fails at
        once instead of waiting.

        Polling never blocks the loop, and a cancelled waiter holds nothing.
        """
        queued = perf_counter()
        STATS.wait(1)
        try:
            while True:
                if len(self._rejected) == len(self.members):
                    raise LLMError(
                        "auth_error",
                        "Every key of the pool is refused (credentials "
                        "revoked or expired)",
                    )
                start = self._next % len(self.members)
                order = [
                    member
                    for member in self.members[start:] + self.members[:start]
                    if self.members.index(member) not in self._rejected
                ]
                candidates = [member for member in order if usable(member)]
                if not candidates:
                    return None
                candidates = [m for m in candidates if self._awake(m)]
                for member in candidates:
                    slot = member._limit()
                    if slot.acquire(blocking=False):
                        # The next request starts with the following key.
                        self._next = self.members.index(member) + 1
                        waited = round((perf_counter() - queued) * 1000)
                        return member, slot, waited
                await asyncio.sleep(0.01)
        finally:
            STATS.wait(-1)

    async def generate(
        self,
        schema: Type[T],
        system: str,
        payload: Dict[str, Any],
        *,
        stage: str = "extract",
    ) -> T:
        _stage(stage)
        route = self.members[0]._route(stage, payload)
        limited = 0

        def usable(member: JsonLLM) -> bool:
            if not member.available(route):
                return False
            if self._embeds(member) is not True:
                return True
            # An embedding key chats only when no other key could.
            return not any(
                self._embeds(other) is not True
                and index not in self._rejected
                and other.available(route)
                for index, other in enumerate(self.members)
            )

        while True:
            held = await self._acquire(usable)
            if held is None:
                raise LLMError(
                    "models_exhausted",
                    f"Every key is out of tokens for the {route} route",
                )
            member, slot, queue_ms = held
            try:
                answer = await member._generate(
                    schema, system, payload, stage, queue_ms
                )
                self._rested(member)
                return answer
            except LLMError as exc:
                if (rate_limited(exc) or rejected_key(exc)) and len(
                    self.members
                ) > 1:
                    # Another key takes it; every key limited in turn is
                    # the caller's retry.
                    self._rest(member, exc)
                    limited += 1
                    if limited >= len(self.members):
                        raise
                    continue
                # This key has no model left; the others may still have.
                if exc.code != "models_exhausted":
                    raise
            finally:
                slot.release()

    async def embed(
        self, texts: Sequence[str], model: str
    ) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        limited = 0
        while True:
            held = await self._acquire(
                lambda member: self._embeds(member) is not False
            )
            if held is None:
                raise LLMError(
                    "models_exhausted",
                    "No key of the pool has embeddings (HTTP 402/403/404)",
                )
            member, slot, queue_ms = held
            try:
                vectors = await member._embed(texts, model, queue_ms)
                self._rested(member)
                self._learn_embeddings(member, True)
                return vectors
            except LLMError as exc:
                if exc.status in (
                    *member.exhausted_statuses,
                    *member.unavailable_statuses,
                ):
                    # This account has no embeddings; the others may.
                    self._learn_embeddings(member, False)
                    logger.info(
                        "Key %s has no embeddings (HTTP %s); not asked again",
                        member.key_name,
                        exc.status,
                    )
                    continue
                if len(self.members) < 2 or not (
                    rate_limited(exc) or rejected_key(exc)
                ):
                    raise
                self._rest(member, exc)
                limited += 1
                if limited >= len(self.members):
                    raise
            finally:
                slot.release()


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
