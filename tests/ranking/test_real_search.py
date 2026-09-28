"""The Docker frontend shows the graph TOP-15, not synthetic data."""

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


def test_docker_build_turns_the_mock_off_before_building():
    dockerfile = (FRONTEND / "Dockerfile").read_text(encoding="utf-8")
    build = dockerfile.index("npm run build")
    flag = re.search(r"VITE_MOCK=0", dockerfile)
    assert flag and flag.start() < build


def test_api_contract_documents_the_real_search_fields():
    api = (FRONTEND / "src" / "api.js").read_text(encoding="utf-8")
    assert "GET /api/search?q=" in api
    for field in ("quotes", "snapshot", "scope", "note", "demo"):
        assert re.search(rf"\b{field}\b", api), field


def test_report_screen_renders_quotes_and_the_scope_note():
    app = (FRONTEND / "src" / "App.jsx").read_text(encoding="utf-8")
    assert "s.quotes" in app
    assert "res.note" in app
