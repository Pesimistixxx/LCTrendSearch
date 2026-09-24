from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


USER_AGENT = "LCTrendSearch/0.1 (+local research project)"


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    request_headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_openalex(work_id: str, mailto: Optional[str] = None) -> Dict[str, Any]:
    identifier = urllib.parse.quote(work_id, safe=":/")
    url = f"https://api.openalex.org/works/{identifier}"
    if mailto:
        url += "?" + urllib.parse.urlencode({"mailto": mailto})
    return fetch_json(url)


def fetch_openalex_page(
    search: str, cursor: str = "*", per_page: int = 100, mailto: Optional[str] = None
) -> Dict[str, Any]:
    params = {"search": search, "cursor": cursor, "per-page": per_page}
    if mailto:
        params["mailto"] = mailto
    return fetch_json("https://api.openalex.org/works?" + urllib.parse.urlencode(params))


def fetch_pypi(package: str) -> Dict[str, Any]:
    return fetch_json(f"https://pypi.org/pypi/{urllib.parse.quote(package, safe='')}/json")


def fetch_pypi_projects() -> List[str]:
    payload = fetch_json(
        "https://pypi.org/simple/",
        {"Accept": "application/vnd.pypi.simple.v1+json"},
    )
    return [str(project["name"]) for project in payload["projects"]]


def fetch_github(repository: str, token: Optional[str] = None) -> Dict[str, Any]:
    repository = repository.strip("/")
    headers = {"X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    base = f"https://api.github.com/repos/{repository}"
    result: Dict[str, Any] = {"repository": fetch_json(base, headers)}
    try:
        result["readme"] = fetch_json(f"{base}/readme", headers)
    except Exception:
        result["readme"] = None
    try:
        result["releases"] = fetch_json(f"{base}/releases?per_page=20", headers)
    except Exception:
        result["releases"] = []
    return result
