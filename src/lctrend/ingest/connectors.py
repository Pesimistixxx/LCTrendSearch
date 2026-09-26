from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..core.config import load_catalog

logger = logging.getLogger(__name__)


def _observed(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {**payload, "_retrieved_at": datetime.now(timezone.utc).isoformat()}


def fetch_json(
    url: str, headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    settings = load_catalog("sources")["http"]
    request_headers = {
        "Accept": "application/json",
        "User-Agent": settings["user_agent"],
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    logger.debug("GET %s", url)
    with urllib.request.urlopen(
        request, timeout=settings["timeout_seconds"]
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_openalex(
    work_id: str, mailto: Optional[str] = None
) -> Dict[str, Any]:
    identifier = urllib.parse.quote(work_id, safe=":/")
    api_base = load_catalog("sources")["platforms"]["openalex"]["api_base"]
    url = f"{api_base}/{identifier}"
    if mailto:
        url += "?" + urllib.parse.urlencode({"mailto": mailto})
    return _observed(fetch_json(url))


def fetch_openalex_page(
    search: str,
    cursor: str = "*",
    per_page: int = 100,
    mailto: Optional[str] = None,
    filter: Optional[str] = None,
) -> Dict[str, Any]:
    params = {"search": search, "cursor": cursor, "per-page": per_page}
    if filter:
        # OpenAlex filter syntax, e.g.
        # "is_oa:true,has_abstract:true,from_publication_date:2024-01-01".
        params["filter"] = filter
    if mailto:
        params["mailto"] = mailto
    payload = fetch_json(
        load_catalog("sources")["platforms"]["openalex"]["api_base"]
        + "?"
        + urllib.parse.urlencode(params)
    )
    observed_at = datetime.now(timezone.utc).isoformat()
    return {
        **payload,
        "results": [
            {**item, "_retrieved_at": observed_at}
            for item in payload.get("results", [])
        ],
    }


def fetch_pdf(url: str) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise ValueError("PDF URL must be HTTP(S)")
    settings = load_catalog("sources")["http"]
    limit = load_catalog("pipeline")["file_limits"]["max_file_bytes"]
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": settings["user_agent"],
        },
    )
    logger.debug("GET PDF %s", url)
    with urllib.request.urlopen(
        request, timeout=settings["timeout_seconds"]
    ) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("PDF exceeds the configured size limit")
    if not raw.startswith(b"%PDF-"):
        # Publishers often answer a PDF link with an HTML landing or
        # captcha page.
        raise ValueError("URL did not return a PDF")
    logger.debug("PDF %s downloaded, %d bytes", url, len(raw))
    return raw


def fetch_pypi(package: str) -> Dict[str, Any]:
    api_base = load_catalog("sources")["platforms"]["pypi"]["api_base"]
    name = urllib.parse.quote(package, safe="")
    return _observed(fetch_json(f"{api_base}/{name}/json"))


def fetch_pypi_projects() -> List[str]:
    settings = load_catalog("sources")["platforms"]["pypi"]
    payload = fetch_json(
        settings["simple_index"], {"Accept": settings["simple_media_type"]}
    )
    return [str(project["name"]) for project in payload["projects"]]


def fetch_github(
    repository: str, token: Optional[str] = None
) -> Dict[str, Any]:
    repository = repository.strip("/")
    settings = load_catalog("sources")["platforms"]["github"]
    headers = {"X-GitHub-Api-Version": settings["api_version"]}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"{settings['api_base']}/{repository}"
    result: Dict[str, Any] = {"repository": fetch_json(base, headers)}
    # Resolve a commit first: a moving default branch cannot identify
    # README content.
    branch = urllib.parse.quote(
        result["repository"].get("default_branch") or "HEAD", safe=""
    )
    result["commit"] = fetch_json(f"{base}/commits/{branch}", headers)
    sha = result["commit"].get("sha")
    if not sha:
        raise ValueError(
            "GitHub commit metadata has no SHA; README cannot be pinned"
        )
    try:
        result["readme"] = fetch_json(
            f"{base}/readme?" + urllib.parse.urlencode({"ref": sha}), headers
        )
    except Exception as exc:
        logger.warning("GitHub %s: README unavailable (%s)", repository, exc)
        result["readme"] = None
    try:
        result["releases"] = fetch_json(
            f"{base}/releases?"
            + urllib.parse.urlencode(
                {"per_page": settings["releases_per_page"]}
            ),
            headers,
        )
    except Exception as exc:
        logger.warning("GitHub %s: releases unavailable (%s)", repository, exc)
        result["releases"] = []
    return _observed(result)
