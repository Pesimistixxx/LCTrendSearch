from lctrend import connectors


def test_openalex_page_includes_search_and_cursor(monkeypatch):
    captured = {}

    def fake_fetch(url, headers=None):
        captured["url"] = url
        return {"results": []}

    monkeypatch.setattr(connectors, "fetch_json", fake_fetch)
    connectors.fetch_openalex_page("edge computing", "next token", 25, "me@example.com")
    assert "search=edge+computing" in captured["url"]
    assert "cursor=next+token" in captured["url"]
    assert "per-page=25" in captured["url"]


def test_pypi_projects_reads_json_simple_index(monkeypatch):
    monkeypatch.setattr(
        connectors,
        "fetch_json",
        lambda url, headers=None: {"projects": [{"name": "one"}, {"name": "two"}]},
    )
    assert connectors.fetch_pypi_projects() == ["one", "two"]
