"""The search-only server of the light image."""

from fastapi.testclient import TestClient

from frontend.server.search_api import allowed_hosts, create_app


class Search:
    def __init__(self):
        self.calls = []

    async def search(self, query, snapshot):
        self.calls.append((query, snapshot))
        return {"query": query, "signals": [], "ranking": "model"}


def test_search_and_health_are_served():
    service = Search()
    client = TestClient(create_app(service), base_url="http://localhost")
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/api/search", params={"q": " финтех "})
    assert response.status_code == 200
    assert response.json()["query"] == "финтех"
    assert service.calls == [("финтех", None)]
    assert client.get("/api/search", params={"q": " "}).status_code == 422
    bad_date = client.get("/api/search", params={"q": "x", "date": "vчера"})
    assert bad_date.status_code == 422


def test_only_search_routes_exist():
    client = TestClient(create_app(Search()), base_url="http://localhost")
    assert client.get("/api/ingest/status").status_code == 404
    assert client.post("/api/ingest/crawls", json={}).status_code in (
        404,
        405,
    )


def test_public_domain_is_allowed(monkeypatch):
    monkeypatch.setenv("LCTREND_DOMAIN", "signals.example.ru")
    monkeypatch.setenv("LCTREND_ALLOWED_HOSTS", "10.0.0.5")
    hosts = allowed_hosts()
    assert "signals.example.ru" in hosts and "10.0.0.5" in hosts
    assert "localhost" in hosts
    client = TestClient(create_app(Search()), base_url="http://evil.test")
    assert client.get("/api/health").status_code == 400
    public = TestClient(
        create_app(Search()), base_url="https://signals.example.ru"
    )
    assert public.get("/api/health").status_code == 200
