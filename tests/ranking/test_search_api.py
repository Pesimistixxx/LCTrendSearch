"""GET /api/search serves the graph TOP-15 in the frontend contract."""

from datetime import date

from fastapi.testclient import TestClient

from frontend.server.app import create_app


class Manager:
    def close(self, wait=False):
        pass


class Search:
    def __init__(self, error=None):
        self.calls, self.error = [], error

    async def search(self, query, snapshot=None):
        self.calls.append((query, snapshot))
        if self.error:
            raise self.error
        return {"query": query, "signals": [], "rejected": [], "stats": {}}


def client(search):
    app = create_app(Manager(), crawl_manager=object(), search_service=search)
    return TestClient(app, base_url="http://localhost")


def test_search_passes_query_and_snapshot_date():
    search = Search()
    with client(search) as http:
        response = http.get(
            "/api/search", params={"q": " финтех ", "date": "2021-01-01"}
        )
    assert response.status_code == 200
    assert response.json()["query"] == "финтех"
    assert search.calls == [("финтех", date(2021, 1, 1))]


def test_search_rejects_an_empty_query_and_a_bad_date():
    search = Search()
    with client(search) as http:
        assert http.get("/api/search", params={"q": " "}).status_code == 422
        assert (
            http.get(
                "/api/search", params={"q": "ai", "date": "yesterday"}
            ).status_code
            == 422
        )
    assert search.calls == []


def test_unavailable_graph_is_a_503_without_internal_details():
    search = Search(RuntimeError("bolt://secret-host refused"))
    with client(search) as http:
        response = http.get("/api/search", params={"q": "ai"})
    assert response.status_code == 503
    assert "secret-host" not in response.text
