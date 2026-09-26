import pytest
from fastapi.testclient import TestClient

from frontend.server.app import create_app


class Manager:
    def __init__(self):
        self.jobs = []
        self.uploads = []
        self.results = {
            ("job", "doc"): {
                "document": {"title": "Статья"},
                "extraction": {"assertions": []},
            }
        }

    def list_jobs(self):
        return self.jobs

    def create_openalex(self, **values):
        job = {"job_id": "job", "status": "queued", **values}
        self.jobs.append(job)
        return job

    def create_files(self, paths, **values):
        self.uploads.extend(paths)
        return {"job_id": "files", "status": "queued", **values}

    def get_job(self, job_id):
        return (
            next((j for j in self.jobs if j["job_id"] == job_id), None)
            or self._missing()
        )

    def _missing(self):
        raise KeyError()

    def cancel_job(self, job_id):
        self.get_job(job_id)["status"] = "cancelling"
        return self.get_job(job_id)

    def get_result(self, job_id, doc_id):
        return self.results[job_id, doc_id]


class Crawls:
    def __init__(self):
        self.crawls = []

    def create(self, topic=""):
        value = {"crawl_id": "crawl", "topic": topic, "status": "queued"}
        self.crawls.append(value)
        return value

    def list_crawls(self):
        return self.crawls

    def get_crawl(self, crawl_id):
        if crawl_id != "crawl" or not self.crawls:
            raise KeyError(crawl_id)
        return self.crawls[0]

    def pause(self, crawl_id):
        self.get_crawl(crawl_id)["status"] = "paused"
        return self.get_crawl(crawl_id)

    def resume(self, crawl_id):
        self.get_crawl(crawl_id)["status"] = "queued"
        return self.get_crawl(crawl_id)

    def retry_failed(self, crawl_id):
        return self.resume(crawl_id)

    def list_materials(self, crawl_id, status=None, limit=100, offset=0):
        self.get_crawl(crawl_id)
        return [{"material_id": "material", "status": status or "pending"}]


@pytest.fixture
def api(tmp_path):
    manager = Manager()
    frontend = tmp_path / "dist"
    frontend.mkdir()
    (frontend / "index.html").write_text(
        "<h1>Сигнал</h1>", encoding="utf-8"
    )
    status = {
        "llm": {"configured": False, "has_key": False},
        "neo4j": {"available": True},
    }
    app = create_app(
        manager,
        crawl_manager=Crawls(),
        frontend_dir=frontend,
        upload_root=tmp_path / "uploads",
        environment_path=tmp_path / ".env",
        status_reader=lambda: status,
    )
    with TestClient(app) as client:
        yield client, manager, tmp_path


def test_real_routes_create_collect_and_cancel_without_model_calls(api):
    client, manager, _ = api
    response = client.post(
        "/api/ingest/jobs", json={"query": "  edge computing  ", "limit": 2}
    )
    assert response.status_code == 202
    assert response.json()["query"] == "edge computing"
    assert response.json()["mode"] == "hybrid"
    assert response.json()["workers"] == 1
    assert client.get("/api/ingest/jobs").json()["jobs"] == manager.jobs
    assert client.get("/api/ingest/jobs/job").json()["status"] == "queued"
    assert (
        client.post("/api/ingest/jobs/job/cancel").json()["status"]
        == "cancelling"
    )


def test_crawl_routes_use_topic_queue_and_preserve_pause_resume(api):
    client, _, _ = api
    response = client.post(
        "/api/ingest/crawls", json={"topic": "  robotics  "}
    )
    assert response.status_code == 202
    assert response.json()["topic"] == "robotics"
    assert (
        client.get("/api/ingest/crawls").json()["crawls"][0] == response.json()
    )
    assert (
        client.post("/api/ingest/crawls/crawl/pause").json()["status"]
        == "paused"
    )
    assert (
        client.post("/api/ingest/crawls/crawl/resume").json()["status"]
        == "queued"
    )
    assert (
        client.post("/api/ingest/crawls/crawl/retry-failed").json()["status"]
        == "queued"
    )
    pending = client.get("/api/ingest/crawls/crawl/materials?status=pending")
    assert pending.json()["materials"][0]["status"] == "pending"
    assert (
        client.get("/api/ingest/crawls/crawl/materials?limit=101").status_code
        == 422
    )
    assert client.get("/api/ingest/crawls/missing").status_code == 404


def test_empty_topic_runs_configured_domains_without_extra_scope_fields(api):
    client, _, _ = api
    assert client.post("/api/ingest/crawls", json={}).json()["topic"] == ""
    assert (
        client.post("/api/ingest/crawls", json={"scope": "all"}).status_code
        == 422
    )


def test_settings_cannot_change_between_crawl_pages(api):
    client, _, _ = api
    client.post("/api/ingest/crawls", json={"topic": "robotics"})
    response = client.post("/api/ingest/settings", json={"model": "new"})
    assert response.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "x", "limit": -1},
        {"query": "x", "workers": 20},
        {"query": "x", "workers": 2},
        {"query": "x", "mode": "search"},
    ],
)
def test_invalid_job_never_reaches_queue(api, body):
    client, manager, _ = api
    assert client.post("/api/ingest/jobs", json=body).status_code == 422
    assert not manager.jobs


def test_status_and_page_are_honest_and_download_is_same_result(api):
    client, manager, _ = api
    assert (
        client.get("/api/ingest/status").json()["llm"]["configured"] is False
    )
    assert "Сигнал" in client.get("/").text
    legacy = client.get("/ingest.html", follow_redirects=False)
    assert legacy.headers["location"] == "/#view=ingest"
    assert client.get("/api/health").json() == {"status": "ok"}
    assert (
        client.get("/api/ingest/jobs/job/documents/doc/result").json()
        == manager.results["job", "doc"]
    )
    download = client.get("/api/ingest/jobs/job/documents/doc/download")
    assert download.json() == manager.results["job", "doc"]
    assert "attachment" in download.headers["Content-Disposition"]
    assert client.get("/api/ingest/jobs/absent").status_code == 404
    assert (
        client.get("/api/ingest/jobs/job/documents/absent/result").status_code
        == 404
    )


def test_upload_preserves_bytes_prevents_path_escape_and_allows_duplicates(
    api,
):
    client, manager, root = api
    response = client.post(
        "/api/ingest/uploads",
        data={"mode": "hybrid", "direction": "Батареи"},
        files=[
            (
                "files",
                (
                    "../../article.md",
                    "Оригинал\n".encode("utf-8"),
                    "text/markdown",
                ),
            ),
            (
                "files",
                (r"C:\secret\article.md", b"Second text", "text/markdown"),
            ),
        ],
    )
    assert response.status_code == 202
    assert len(manager.uploads) == 2
    assert manager.uploads[0].read_bytes() == "Оригинал\n".encode("utf-8")
    assert manager.uploads[1].read_bytes() == b"Second text"
    assert len({p.name.casefold() for p in manager.uploads}) == 2
    assert all(p.is_relative_to(root / "uploads") for p in manager.uploads)


@pytest.mark.parametrize(
    "name,content", [("bad.exe", b"data"), ("empty.md", b"")]
)
def test_upload_rejects_invalid_files_and_removes_only_new_bytes(
    api, name, content
):
    client, manager, root = api
    assert (
        client.post(
            "/api/ingest/uploads", files={"files": (name, content)}
        ).status_code
        == 400
    )
    assert not manager.uploads
    assert not list((root / "uploads").glob("**/*.*"))


def test_cross_origin_form_cannot_launch_paid_job(api):
    client, manager, _ = api
    assert (
        client.post(
            "/api/ingest/jobs",
            json={"query": "x"},
            headers={"Origin": "https://elsewhere.example"},
        ).status_code
        == 403
    )
    assert not manager.jobs


def test_proxy_preserves_same_origin_on_custom_frontend_port(api):
    client, manager, _ = api
    response = client.post(
        "/api/ingest/jobs",
        json={"query": "robotics"},
        headers={
            "Host": "localhost:9000",
            "Origin": "http://localhost:9000",
        },
    )
    assert response.status_code == 202
    assert manager.jobs[0]["query"] == "robotics"


def test_settings_validate_locally_preserve_secret_and_do_not_return_it(
    api, monkeypatch
):
    client, manager, root = api
    for key in (
        "LLM_MODEL",
        "LLM_EXTRACT_MODEL",
        "LLM_REVIEW_MODEL",
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    response = client.post(
        "/api/ingest/settings",
        json={
            "model": "local-model",
            "base_url": "http://127.0.0.1:1234/v1",
            "api_key": "user-secret",
        },
    )
    assert response.status_code == 200
    assert "user-secret" not in response.text
    assert "user-secret" in (root / ".env").read_text(encoding="utf-8")
    # A blank password input means keeping the existing secret.
    assert (
        client.post(
            "/api/ingest/settings",
            json={
                "model": "other-local",
                "base_url": "http://127.0.0.1:1234/v1",
                "api_key": "",
            },
        ).status_code
        == 200
    )
    assert "user-secret" in (root / ".env").read_text(encoding="utf-8")
    manager.jobs.append({"status": "running"})
    assert (
        client.post("/api/ingest/settings", json={"model": "x"}).status_code
        == 409
    )


def test_bad_model_configuration_is_rolled_back_without_saving(
    api, monkeypatch
):
    client, _, root = api
    monkeypatch.setenv("LLM_MODEL", "original")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    import os

    assert (
        client.post(
            "/api/ingest/settings",
            json={"model": "new", "base_url": "https://provider.example/v1"},
        ).status_code
        == 400
    )
    assert os.getenv("LLM_MODEL") == "original"
    assert not (root / ".env").exists()


def test_container_ui_writes_settings_to_persistent_directory(
    tmp_path, monkeypatch
):
    import os

    from lctrend.core.config import load_environment

    saved = tmp_path / "artifacts" / "ingestion" / "settings.env"
    monkeypatch.setenv("LCTREND_SETTINGS_FILE", str(saved))
    for key in (
        "LLM_PROVIDER", "LLM_MODEL", "LLM_EXTRACT_MODEL",
        "LLM_REVIEW_MODEL", "LLM_BASE_URL", "LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    app = create_app(Manager(), status_reader=lambda: {"ok": True})
    with TestClient(app) as client:
        response = client.post(
            "/api/ingest/settings",
            json={
                "model": "saved-model",
                "base_url": "http://localhost:1234/v1",
                "api_key": "saved-secret",
            },
        )
    assert response.status_code == 200
    assert saved.is_file()
    assert "saved-secret" not in response.text
    # A recreated container receives the original env_file again.
    monkeypatch.setenv("LLM_MODEL", "original-model")
    monkeypatch.setenv("LLM_API_KEY", "original-key")
    load_environment(tmp_path / "missing.env")
    assert os.environ["LLM_MODEL"] == "saved-model"
    assert os.environ["LLM_API_KEY"] == "saved-secret"
