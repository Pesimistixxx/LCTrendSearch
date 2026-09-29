"""Page through source search results without selecting or ranking trends.

These adapters report source-imposed coverage limits explicitly. Discovery
records are identifiers and metadata; GitHub and linked PyPI records still
need their normal native connector before parsing.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any, Mapping
from urllib.parse import unquote, urlencode, urlsplit

from ..core.aio import resolve
from ..core.config import load_catalog
from .connectors import (
    fetch_json,
    fetch_openalex_page,
    normalize_openalex_work_id,
)


def _topic(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Discovery topic must be a nonempty string")
    return value.strip()


def _count(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        else None
    )


def _doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if parsed.hostname not in {"doi.org", "dx.doi.org", "www.doi.org"}:
            return None
        value = parsed.path.lstrip("/")
    else:
        value = re.sub(r"^doi\s*:\s*", "", value, flags=re.IGNORECASE)
    value = unquote(value).strip().casefold()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def _package(value: str) -> str:
    parsed = (
        urlsplit(value)
        if value.lower().startswith(("http://", "https://"))
        else None
    )
    if parsed:
        if parsed.hostname not in {"pypi.org", "www.pypi.org"}:
            raise ValueError("Invalid PyPI project URL")
        parts = parsed.path.strip("/").split("/")
        if len(parts) < 2 or parts[0] != "project":
            raise ValueError("Invalid PyPI project URL")
        value = parts[1]
    value = unquote(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", value):
        raise ValueError("Invalid PyPI project name")
    return re.sub(r"[-_.]+", "-", value).casefold()


def _repository(value: str) -> str:
    value = str(value).strip()
    if value.lower().startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if parsed.hostname not in {
            "github.com",
            "www.github.com",
            "api.github.com",
        }:
            raise ValueError("Invalid GitHub repository URL")
        value = parsed.path.strip("/")
        if parsed.hostname == "api.github.com" and value.startswith("repos/"):
            value = value[6:]
        value = "/".join(value.split("/")[:2])
    value = unquote(value).strip("/")
    if value.lower().endswith(".git"):
        value = value[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("Invalid GitHub repository identifier")
    return value.casefold()


def material_identity(
    source: str, source_id: str, payload: Mapping | None = None
) -> str:
    """Stable material identity across DOI URL forms and source case aliases.

    This is document deduplication, not a claim about shared independence of
    repositories, packages or papers. Different material types stay distinct.
    """
    source = source.strip().casefold()
    payload = payload or {}
    ids = payload.get("ids", {})
    doi = _doi(payload.get("doi")) or (
        _doi(ids.get("doi")) if isinstance(ids, Mapping) else None
    )
    # Repository/package DOI metadata may describe an accompanying paper.
    # Only article sources can use it as the identity of this material.
    if doi and source in {"openalex", "crossref"}:
        return f"doi:{doi}"
    if source == "github":
        repo = payload.get("repository", payload)
        value = repo.get("full_name") if isinstance(repo, Mapping) else None
        return "github:" + _repository(value or source_id)
    if source == "pypi":
        info = payload.get("info", {})
        value = info.get("name") if isinstance(info, Mapping) else None
        return "pypi:" + _package(value or source_id)
    if source == "openalex":
        value = normalize_openalex_work_id(
            str(payload.get("id") or source_id)
        )
        if value.startswith("doi:"):
            return value
        return "openalex:" + value.casefold()
    if not source or not str(source_id).strip():
        raise ValueError("Material source and identifier are required")
    return f"{source}:{str(source_id).strip()}"


async def discover_openalex(topic: str, cursor: str | None = "*") -> dict:
    """Fetch one cursor page from OpenAlex, preserving the source's total."""
    payload = await resolve(
        fetch_openalex_page(
            _topic(topic),
            cursor or "*",
            100,
            os.getenv("OPENALEX_MAILTO"),
            None,
        )
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid OpenAlex discovery response")
    records, meta = payload.get("results"), payload.get("meta", {})
    if not isinstance(records, list) or not isinstance(meta, Mapping):
        raise ValueError("Invalid OpenAlex discovery response")
    items = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Invalid OpenAlex discovery record")
        source_id = str(record.get("id") or record.get("doi") or "")
        identity = material_identity("openalex", source_id, record)
        location = record.get("primary_location") or {}
        landing = (
            location.get("landing_page_url")
            if isinstance(location, Mapping)
            else None
        )
        items.append(
            {
                "source": "openalex",
                "source_id": source_id,
                "canonical_id": identity,
                "title": str(
                    record.get("title")
                    or record.get("display_name")
                    or source_id
                ),
                "url": record.get("doi") or landing or source_id,
                "payload": dict(record),
            }
        )
    if items and "next_cursor" not in meta:
        raise ValueError("OpenAlex response is missing next_cursor")
    next_cursor = meta.get("next_cursor")
    if next_cursor is not None and (
        not isinstance(next_cursor, str) or not next_cursor
    ):
        raise ValueError("Invalid OpenAlex cursor")
    if items and next_cursor == (cursor or "*"):
        raise ValueError("Repeated OpenAlex cursor")
    # An empty page terminates accessible paging even if a source supplied a
    # stale next cursor; expose the inconsistency rather than loop forever.
    limitations = []
    if not items and next_cursor:
        limitations.append(
            {
                "code": "empty_page_with_cursor",
                "message": (
                    "Источник вернул пустую страницу с курсором продолжения."
                ),
            }
        )
        next_cursor = None
    return {
        "items": items,
        "next_cursor": next_cursor,
        "total": _count(meta.get("count")),
        "complete": next_cursor is None and not limitations,
        "limitations": limitations,
    }


async def discover_github(topic: str, cursor: str | None = None) -> dict:
    """Fetch repository search pages; never claim coverage beyond the API cap.

    GitHub permits at most 1,000 matches per query:
    https://docs.github.com/en/rest/search/search
    """
    topic = _topic(topic)
    page = 1 if cursor in (None, "", "*") else int(cursor)
    if not 1 <= page <= 10:
        raise ValueError("GitHub discovery page must be 1..10")
    settings = load_catalog("sources")["platforms"]["github"]
    base = settings["api_base"].rstrip("/")
    if not base.endswith("/repos"):
        raise ValueError("GitHub API base must end in /repos")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": settings["api_version"],
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = await resolve(
        fetch_json(
            base[:-6]
            + "/search/repositories?"
            + urlencode({"q": topic, "per_page": 100, "page": page}),
            headers,
        )
    )
    records = payload.get("items")
    if not isinstance(records, list):
        raise ValueError("Invalid GitHub discovery response")
    items = []
    catalog = re.compile(settings["catalog_pattern"])
    for record in records:
        if not isinstance(record, Mapping) or not record.get("full_name"):
            raise ValueError("Invalid GitHub discovery record")
        if catalog.search(str(record.get("name") or "")) or catalog.search(
            str(record.get("description") or "")
        ):
            # A link catalog lists libraries; it is not a technology source.
            continue
        source_id = _repository(record["full_name"])
        items.append(
            {
                "source": "github",
                "source_id": source_id,
                "canonical_id": material_identity("github", source_id, record),
                "title": str(record["full_name"]),
                "url": record.get("html_url")
                or settings["public_base"] + "/" + source_id,
                "payload": dict(record),
            }
        )
    reported = _count(payload.get("total_count"))
    total = min(reported, 1000) if reported is not None else None
    capped, incomplete = (
        reported is not None and reported > 1000,
        bool(payload.get("incomplete_results")),
    )
    limitations = []
    if capped:
        limitations.append(
            {
                "code": "github_search_cap",
                "reported_total": reported,
                "accessible_total": 1000,
                "message": (
                    "GitHub отдаёт максимум 1000 репозиториев на один запрос; "
                    "весь спектр этим запросом не покрыт."
                ),
            }
        )
    if incomplete:
        limitations.append(
            {
                "code": "github_incomplete_results",
                "message": "GitHub сообщил о неполной выдаче запроса.",
            }
        )
    has_more = (
        bool(items)
        and page < 10
        and (page * 100 < total if total is not None else len(items) == 100)
    )
    if not items and total is not None and (page - 1) * 100 < total:
        limitations.append(
            {
                "code": "empty_page_before_total",
                "message": (
                    "GitHub завершил доступную выдачу "
                    "раньше заявленного количества."
                ),
            }
        )
    if page == 10 and reported is None and len(items) == 100:
        limitations.append(
            {
                "code": "github_search_cap",
                "accessible_total": 1000,
                "message": (
                    "Достигнут предел 1000 результатов; "
                    "общее количество источник не сообщил."
                ),
            }
        )
    return {
        "items": items,
        "next_cursor": str(page + 1) if has_more else None,
        "total": total,
        "complete": not has_more and not limitations,
        "limitations": limitations,
    }


def discover_pypi_from_github_payload(payload: Mapping) -> list[dict]:
    """Discover only explicitly linked PyPI packages in a fetched README.

    There is no invented PyPI thematic search and no inferred package name.
    """
    readme = payload.get("readme")
    if not isinstance(readme, Mapping):
        return []
    text = readme.get("text")
    if not isinstance(text, str) and readme.get("content"):
        try:
            text = base64.b64decode(readme["content"]).decode(
                "utf-8", errors="replace"
            )
        except (ValueError, TypeError):
            return []
    if not isinstance(text, str):
        return []
    items = {}
    for match in re.finditer(
        r"https?://(?:www\.)?pypi\.org/project/([A-Za-z0-9_.%~-]+)",
        text,
        re.IGNORECASE,
    ):
        try:
            # A sentence may end with a bare project URL and a full stop;
            # PyPI project names cannot end in a dot.
            name = _package(match.group(1).rstrip("."))
        except ValueError:
            continue
        items.setdefault(
            name,
            {
                "source": "pypi",
                "source_id": name,
                "canonical_id": "pypi:" + name,
                "title": name,
                "url": "https://pypi.org/project/" + name + "/",
                "payload": None,
            },
        )
    return list(items.values())
