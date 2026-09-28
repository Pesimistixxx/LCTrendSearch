"""Asynchronous source API clients with bounded retries.

Transient failures (timeouts, 429, 5xx) and GitHub rate limits are retried
with exponential backoff that honours Retry-After and X-RateLimit-Reset.
Permanent errors (404, 401, malformed data) are raised immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import threading
import time
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import httpx

from ..core.aio import resolve
from ..core.config import load_catalog

logger = logging.getLogger(__name__)

# Tests inject an httpx.MockTransport here; production uses the network.
TRANSPORT: Optional[httpx.AsyncBaseTransport] = None
# Without an NCBI API key PubMed allows three requests per second per
# process. Web jobs run in their own event loops, so an asyncio.Lock bound
# to the first loop would fail in the next one (NEW-3): request start times
# are reserved under a thread lock instead.
_PUBMED_INTERVAL_SECONDS = 0.4
_PUBMED_SLOT_LOCK = threading.Lock()
_pubmed_next_slot = 0.0


async def _pubmed_slot() -> None:
    global _pubmed_next_slot
    with _PUBMED_SLOT_LOCK:
        now = time.monotonic()
        start = max(now, _pubmed_next_slot)
        _pubmed_next_slot = start + _PUBMED_INTERVAL_SECONDS
    await asyncio.sleep(start - now)


# Host -> (monotonic time its rate limit resets, the refusing status). A
# host that asked to wait longer than max_rate_limit_wait_seconds is not
# asked again until then: a GitHub crawl of hundreds of repositories would
# otherwise spend a request and a warning on each one.
_HOST_BLOCKED: Dict[str, tuple] = {}
_HOST_BLOCKED_LOCK = threading.Lock()


def _blocked(url: str) -> Optional[int]:
    """The refusing status while the host's rate limit lasts, else None."""
    host = _host(url)
    with _HOST_BLOCKED_LOCK:
        until, status = _HOST_BLOCKED.get(host, (0.0, 0))
        if until > time.monotonic():
            return status
        _HOST_BLOCKED.pop(host, None)
    return None


def _block(url: str, status: int, delay: float) -> None:
    with _HOST_BLOCKED_LOCK:
        _HOST_BLOCKED[_host(url)] = (time.monotonic() + delay, status)


class SourceHTTPError(RuntimeError):
    """A source answered with a non-retryable (or exhausted) HTTP error."""

    def __init__(self, status: int, url: str):
        self.status = status
        self.code = f"http_{status}"
        super().__init__(f"HTTP {status} from {_host(url)}")


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or "source"


def _observed(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {**payload, "_retrieved_at": datetime.now(timezone.utc).isoformat()}


def _retry_after(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                moment = parsedate_to_datetime(value)
                if moment.tzinfo is None:
                    moment = moment.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (moment - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return None
    # GitHub primary rate limit: 403/429 with remaining=0 and a reset time.
    if response.headers.get("X-RateLimit-Remaining") == "0":
        try:
            reset = float(response.headers["X-RateLimit-Reset"])
        except (KeyError, ValueError):
            return None
        # OpenAlex reports seconds until reset; GitHub reports a Unix time.
        if response.request.url.host == "api.openalex.org":
            return max(0.0, reset) + 1.0
        return max(0.0, reset - time.time()) + 1.0
    return None


def _rate_limited(response: httpx.Response) -> bool:
    # Primary limit: remaining=0. Secondary (abuse) limit: 403 with
    # Retry-After and requests still remaining (A-12).
    return response.status_code == 403 and (
        response.headers.get("X-RateLimit-Remaining") == "0"
        or "Retry-After" in response.headers
    )


async def request(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    *,
    max_bytes: Optional[int] = None,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    """GET (or an idempotent search POST) with retries; the body is read
    (and bounded by ``max_bytes``).
    """
    settings = load_catalog("sources")["http"]
    request_headers = {"User-Agent": settings["user_agent"]}
    request_headers.update(headers or {})
    attempts = int(settings.get("max_attempts", 1))
    base = float(settings.get("backoff_seconds", 1.0))
    cap = float(settings.get("max_backoff_seconds", 60.0))
    limit_wait = float(settings.get("max_rate_limit_wait_seconds", cap))
    retry_statuses = set(settings.get("retry_statuses", []))
    blocked = _blocked(url)
    if blocked is not None:
        raise SourceHTTPError(blocked, url)
    async with httpx.AsyncClient(
        timeout=settings["timeout_seconds"],
        transport=TRANSPORT,
        follow_redirects=True,
    ) as client:
        for attempt in range(1, attempts + 1):
            delay: Optional[float] = None
            try:
                async with client.stream(
                    method, url, headers=request_headers, json=json_body
                ) as response:
                    if response.is_success:
                        body = bytearray()
                        async for part in response.aiter_bytes():
                            body.extend(part)
                            if max_bytes is not None and len(body) > max_bytes:
                                raise ValueError(
                                    "Response exceeds the configured size "
                                    "limit"
                                )
                        # aiter_bytes() already decoded the wire body. Drop
                        # its encoding and length before rebuilding a response.
                        decoded_headers = httpx.Headers(response.headers)
                        decoded_headers.pop("Content-Encoding", None)
                        decoded_headers.pop("Content-Length", None)
                        return httpx.Response(
                            response.status_code,
                            headers=decoded_headers,
                            content=bytes(body),
                            request=response.request,
                        )
                    status = response.status_code
                    if status in retry_statuses or _rate_limited(response):
                        delay = _retry_after(response)
                        if delay is not None and delay > limit_wait:
                            logger.warning(
                                "%s asks to wait %.0fs; giving up, no "
                                "requests to it until then",
                                _host(url),
                                delay,
                            )
                            _block(url, status, delay)
                            raise SourceHTTPError(status, url)
                    else:
                        raise SourceHTTPError(status, url)
                    if attempt == attempts:
                        raise SourceHTTPError(status, url)
                    reason = f"HTTP {status}"
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == attempts:
                    raise
                reason = type(exc).__name__
            if delay is None:
                delay = min(cap, base * 2 ** (attempt - 1))
                delay += random.uniform(0, delay / 4)
            logger.warning(
                "%s %s failed (%s), retry %d/%d in %.1fs",
                method,
                _host(url),
                reason,
                attempt,
                attempts - 1,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def fetch_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """GET a JSON document, or POST ``json_body`` to a search endpoint."""
    method = "GET" if json_body is None else "POST"
    logger.debug("%s %s", method, url)
    response = await request(
        url,
        {"Accept": "application/json", **(headers or {})},
        method=method,
        json_body=json_body,
    )
    return json.loads(response.content.decode("utf-8"))


def normalize_openalex_work_id(work_id: str) -> str:
    """Normalize a work key, OpenAlex URL or DOI to a singleton API ID."""
    value = work_id.strip()
    if value.lower().startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(value)
        if parsed.username or parsed.password or parsed.port:
            raise ValueError("Invalid OpenAlex work URL")
        host = (parsed.hostname or "").lower()
        path = urllib.parse.unquote(parsed.path).strip("/")
        if host in {"openalex.org", "www.openalex.org", "api.openalex.org"}:
            value = path.removeprefix("works/")
        elif host in {"doi.org", "dx.doi.org"}:
            value = "doi:" + path
        else:
            raise ValueError("Work URL must use OpenAlex or doi.org")
    value = value.removeprefix("works/")
    if re.fullmatch(r"W\d+", value, re.I):
        return value.upper()
    if value.lower().startswith("doi:"):
        value = value[4:]
    if re.fullmatch(r"10\.\d+/\S+", value):
        return "doi:" + value.casefold()
    if re.fullmatch(r"pmid:\d+", value, re.I):
        return value.lower()
    raise ValueError("Expected an OpenAlex work ID (W...), DOI or work URL")


def _openalex_headers(api_key: Optional[str]) -> Optional[Dict[str, str]]:
    key = os.getenv("OPENALEX_API_KEY") if api_key is None else api_key
    key = (key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else None


async def _fetch_openalex_json(
    url: str, api_key: Optional[str]
) -> Dict[str, Any]:
    headers = _openalex_headers(api_key)
    # Keep unauthenticated calls compatible with simple injected fetchers.
    if headers:
        return await resolve(fetch_json(url, headers))
    return await resolve(fetch_json(url))


async def fetch_openalex(
    work_id: str,
    mailto: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    identifier = urllib.parse.quote(
        normalize_openalex_work_id(work_id), safe=":/"
    )
    api_base = load_catalog("sources")["platforms"]["openalex"]["api_base"]
    url = f"{api_base}/{identifier}"
    if mailto:
        url += "?" + urllib.parse.urlencode({"mailto": mailto})
    payload = await _fetch_openalex_json(url, api_key)
    if not isinstance(payload, dict):
        raise ValueError("OpenAlex response must contain a work object")
    return _observed(payload)


async def fetch_openalex_page(
    search: str,
    cursor: str = "*",
    per_page: int = 100,
    mailto: Optional[str] = None,
    filter: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if isinstance(per_page, bool) or not isinstance(per_page, int):
        raise ValueError("OpenAlex per_page must be an integer from 1 to 100")
    if not 1 <= per_page <= 100:
        raise ValueError("OpenAlex per_page must be from 1 to 100")
    params = {"search": search, "cursor": cursor, "per-page": per_page}
    if filter:
        # OpenAlex filter syntax, e.g.
        # "is_oa:true,has_abstract:true,from_publication_date:2024-01-01".
        params["filter"] = filter
    if mailto:
        params["mailto"] = mailto
    payload = await _fetch_openalex_json(
        load_catalog("sources")["platforms"]["openalex"]["api_base"]
        + "?"
        + urllib.parse.urlencode(params),
        api_key,
    )
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("results"), list)
        or any(not isinstance(item, dict) for item in payload["results"])
    ):
        raise ValueError("OpenAlex response must contain a list of works")
    observed_at = datetime.now(timezone.utc).isoformat()
    return {
        **payload,
        "results": [
            {**item, "_retrieved_at": observed_at}
            for item in payload.get("results", [])
        ],
    }


async def fetch_pdf(url: str) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise ValueError("PDF URL must be HTTP(S)")
    limit = load_catalog("pipeline")["file_limits"]["max_file_bytes"]
    logger.debug("GET PDF %s", url)
    try:
        response = await request(
            url, {"Accept": "application/pdf"}, max_bytes=limit
        )
    except ValueError:
        raise ValueError("PDF exceeds the configured size limit") from None
    raw = response.content
    if not raw.startswith(b"%PDF-"):
        # Publishers often answer a PDF link with an HTML landing or
        # captcha page.
        raise ValueError("URL did not return a PDF")
    logger.debug("PDF %s downloaded, %d bytes", url, len(raw))
    return raw


async def fetch_pubmed_xml(pmid: str) -> bytes:
    """Fetch one PubMed record from a fixed endpoint using a numeric PMID."""
    if not re.fullmatch(r"[1-9][0-9]{0,11}", pmid):
        raise ValueError("PubMed PMID must be numeric")
    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        + urllib.parse.urlencode(
            {"db": "pubmed", "id": pmid, "retmode": "xml"}
        )
    )
    logger.debug("GET PubMed PMID %s", pmid)
    await _pubmed_slot()
    response = await request(
        url, {"Accept": "application/xml"}, max_bytes=1_000_000
    )
    return response.content


async def fetch_pypi(package: str) -> Dict[str, Any]:
    api_base = load_catalog("sources")["platforms"]["pypi"]["api_base"]
    name = urllib.parse.quote(package, safe="")
    return _observed(await resolve(fetch_json(f"{api_base}/{name}/json")))


async def fetch_pypi_projects() -> List[str]:
    settings = load_catalog("sources")["platforms"]["pypi"]
    payload = await resolve(
        fetch_json(
            settings["simple_index"],
            {"Accept": settings["simple_media_type"]},
        )
    )
    return [str(project["name"]) for project in payload["projects"]]


async def fetch_github(
    repository: str, token: Optional[str] = None
) -> Dict[str, Any]:
    repository = repository.strip("/")
    settings = load_catalog("sources")["platforms"]["github"]
    headers = {"X-GitHub-Api-Version": settings["api_version"]}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"{settings['api_base']}/{repository}"
    result: Dict[str, Any] = {
        "repository": await resolve(fetch_json(base, headers))
    }
    # Resolve a commit first: a moving default branch cannot identify
    # README content.
    branch = urllib.parse.quote(
        result["repository"].get("default_branch") or "HEAD", safe=""
    )
    result["commit"] = await resolve(
        fetch_json(f"{base}/commits/{branch}", headers)
    )
    sha = result["commit"].get("sha")
    if not sha:
        raise ValueError(
            "GitHub commit metadata has no SHA; README cannot be pinned"
        )
    readme_url = f"{base}/readme?" + urllib.parse.urlencode({"ref": sha})
    releases_url = f"{base}/releases?" + urllib.parse.urlencode(
        {"per_page": settings["releases_per_page"]}
    )
    # Weekly statistics are dated history (commit cadence, first commit of
    # each contributor), so snapshot features stay point-in-time. GitHub
    # answers 202 with an empty body while it computes them; the repository
    # is still usable without them.
    readme, releases, activity, contributors = await asyncio.gather(
        resolve(fetch_json(readme_url, headers)),
        resolve(fetch_json(releases_url, headers)),
        resolve(fetch_json(f"{base}/stats/commit_activity", headers)),
        resolve(fetch_json(f"{base}/stats/contributors", headers)),
        return_exceptions=True,
    )
    if isinstance(readme, BaseException):
        logger.warning(
            "GitHub %s: README unavailable (%s)", repository, readme
        )
        readme = None
    if isinstance(releases, BaseException):
        logger.warning(
            "GitHub %s: releases unavailable (%s)", repository, releases
        )
        releases = None
    for name, value in (
        ("commit activity", activity),
        ("contributors", contributors),
    ):
        if isinstance(value, BaseException):
            logger.warning(
                "GitHub %s: %s unavailable (%s)", repository, name, value
            )
    result["readme"], result["releases"] = readme, releases
    result["commit_activity"] = (
        activity if isinstance(activity, list) else None
    )
    result["contributor_stats"] = (
        contributors if isinstance(contributors, list) else None
    )
    return _observed(result)
