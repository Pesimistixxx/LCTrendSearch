import asyncio

from lctrend.ingest import connectors


def test_openalex_page_includes_search_and_cursor(monkeypatch):
    captured = {}

    def fake_fetch(url, headers=None):
        captured["url"] = url
        return {"results": []}

    monkeypatch.setattr(connectors, "fetch_json", fake_fetch)
    asyncio.run(
        connectors.fetch_openalex_page(
            "edge computing", "next token", 25, "me@example.com"
        )
    )
    assert "search=edge+computing" in captured["url"]
    assert "cursor=next+token" in captured["url"]
    assert "per-page=25" in captured["url"]


def test_pypi_projects_reads_json_simple_index(monkeypatch):
    monkeypatch.setattr(
        connectors,
        "fetch_json",
        lambda url, headers=None: {
            "projects": [{"name": "one"}, {"name": "two"}]
        },
    )
    assert asyncio.run(connectors.fetch_pypi_projects()) == ["one", "two"]


def test_github_fetches_commit_then_reads_same_sha(monkeypatch):
    calls = []

    def fake_fetch(url, headers=None):
        calls.append(url)
        if url.endswith("/repos/org/repo"):
            return {"full_name": "org/repo", "default_branch": "feature/main"}
        if "/commits/" in url:
            return {
                "sha": "fixedsha",
                "commit": {"committer": {"date": "2025-01-01"}},
            }
        if "/readme?" in url:
            return {"text": "README"}
        return []

    monkeypatch.setattr(connectors, "fetch_json", fake_fetch)
    payload = asyncio.run(connectors.fetch_github("org/repo"))
    assert calls[1].endswith("/commits/feature%2Fmain")
    assert calls[2].endswith("/readme?ref=fixedsha")
    assert payload["commit"]["sha"] == "fixedsha"
    assert payload["_retrieved_at"]
