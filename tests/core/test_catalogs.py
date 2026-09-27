import json

import pytest

from lctrend.core.config import (
    cypher_identifier,
    load_catalog,
    load_environment,
    resource_path,
)


def test_catalog_override_is_separate_data_and_invalid_override_is_explicit(
    tmp_path, monkeypatch
):
    (tmp_path / "resolver.json").write_text(
        json.dumps({"explicit_aliases": []}), encoding="utf-8"
    )
    monkeypatch.setenv("LCTREND_CONFIG_DIR", str(tmp_path))
    assert load_catalog("resolver") == {"explicit_aliases": []}
    assert load_catalog("runtime")["openalex"]["per_page"] == 100
    (tmp_path / "resolver.json").write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot load catalog"):
        load_catalog("resolver")


def test_schema_identifiers_and_catalog_names_cannot_inject_paths_or_cypher():
    assert cypher_identifier("SUBJECT") == "SUBJECT"
    with pytest.raises(ValueError):
        cypher_identifier("TASK) DELETE n")
    with pytest.raises(ValueError):
        resource_path("../secrets")


def test_container_settings_survive_restart_and_only_override_allowed_keys(
    tmp_path, monkeypatch
):
    import os

    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=initial\n", encoding="utf-8")
    settings = tmp_path / "settings.env"
    settings.write_text(
        "LLM_MODEL=saved\nLLM_API_KEY=saved-key\n"
        "OPENALEX_API_KEY=saved-openalex\n"
        "OPENALEX_MAILTO=research@example.org\nNEO4J_URI=wrong\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LCTREND_SETTINGS_FILE", str(settings))
    monkeypatch.setenv("LLM_MODEL", "initial")
    monkeypatch.setenv("LLM_API_KEY", "initial-key")
    monkeypatch.setenv("OPENALEX_API_KEY", "initial-openalex")
    monkeypatch.setenv("OPENALEX_MAILTO", "initial@example.org")
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://external.example")
    load_environment(env_file)
    assert os.environ["LLM_MODEL"] == "saved"
    assert os.environ["LLM_API_KEY"] == "saved-key"
    assert os.environ["OPENALEX_API_KEY"] == "saved-openalex"
    assert os.environ["OPENALEX_MAILTO"] == "research@example.org"
    assert os.environ["NEO4J_URI"] == "neo4j+s://external.example"
