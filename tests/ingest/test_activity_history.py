import asyncio

from lctrend.ingest import connectors
from lctrend.ingest.adapters import parse_github, parse_pypi


def test_unavailable_stats_are_missing_instead_of_zero(monkeypatch):
    async def fetch(url, headers=None):
        if url.endswith("/repos/org/repo"):
            return {"full_name": "org/repo", "default_branch": "main"}
        if "/commits/" in url:
            return {
                "sha": "a",
                "commit": {"committer": {"date": "2020-01-01"}},
            }
        if "/readme?" in url:
            return {"text": "Technology"}
        raise RuntimeError("statistics unavailable")

    monkeypatch.setattr(connectors, "fetch_json", fetch)
    payload = asyncio.run(connectors.fetch_github("org/repo"))
    document = parse_github(payload)
    assert "commit_weeks" not in document.metadata
    assert "contributor_first_weeks" not in document.metadata
    assert "release_dates" not in document.metadata


def test_empty_collected_stats_remain_known_empty():
    document = parse_github(
        {
            "full_name": "org/repo",
            "commit_activity": [],
            "contributor_stats": [],
            "releases": [],
        }
    )
    assert document.metadata["commit_weeks"] == {}
    assert document.metadata["contributor_first_weeks"] == []
    assert document.metadata["release_dates"] == []


def test_pypi_first_release_dates_project_but_current_release_dates_content():
    document = parse_pypi(
        {
            "info": {"name": "example", "version": "2"},
            "urls": [{"upload_time_iso_8601": "2025-01-01"}],
            "releases": {
                "1": [
                    {"upload_time_iso_8601": "2020-01-01"},
                    {"upload_time_iso_8601": "2020-01-02"},
                ],
                "2": [{"upload_time_iso_8601": "2025-01-01"}],
            },
        }
    )
    assert document.published_at == "2020-01-01"
    assert document.version_published_at == "2025-01-01"
    assert document.metadata["release_dates"] == ["2020-01-01", "2025-01-01"]
