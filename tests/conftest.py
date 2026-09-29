import logging

import pytest

from lctrend.core.logging_config import remove_handlers


@pytest.fixture(autouse=True)
def in_process_docling(monkeypatch):
    """Fake Docling modules live only in the test process; tests of the
    isolated converter process switch this back off.
    """
    from lctrend.ingest import file_adapters

    monkeypatch.setattr(file_adapters, "IN_PROCESS_DOCLING", True)


@pytest.fixture(autouse=True)
def isolated_log_file(monkeypatch, tmp_path):
    """Keep CLI runs from writing to the project's logs/ directory."""
    monkeypatch.setenv("LCTREND_LOG_FILE", str(tmp_path / "lctrend.log"))
    # Offline tests never call the embeddings API; tests of the semantic
    # layer inject a fake deduplicator.
    monkeypatch.setenv("DEDUP_IN_LLM", "0")
    # A developer's .env (loaded into the process by CLI tests) must never
    # reach offline tests: its key pool, provider or Docker-only CA paths.
    for name in (
        "GIGACHAT_KEYS_FILE",
        "LLM_PROVIDER",
        "GIGACHAT_CA_BUNDLE_FILE",
        "LLM_CA_BUNDLE_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    for name in ("lctrend", "frontend.server"):
        logger = logging.getLogger(name)
        remove_handlers(logger)
        logger.propagate = True


@pytest.fixture(autouse=True)
def forget_embedding_keys():
    """Which keys embed, rest or block a host is process-wide; each test
    starts from nothing."""
    from lctrend.ingest import connectors
    from lctrend.llm.client import reset_key_knowledge

    reset_key_knowledge()
    connectors._HOST_BLOCKED.clear()
    yield
    reset_key_knowledge()
    connectors._HOST_BLOCKED.clear()


@pytest.fixture(autouse=True)
def technology_contract_mode(request, monkeypatch):
    """Modules written before the technology contract
    (docs/technology-contract.md) build minimal Technology entities to test
    other stages; they mark themselves ``legacy_technology_entities`` and
    run with the contract reported, not enforced. Contract tests and all
    other modules run it enforced, whatever the shipped switch says, so the
    demotion stays tested; ``shipped_technology_contract`` keeps the
    catalog's own setting.
    """
    if request.node.get_closest_marker("shipped_technology_contract"):
        return
    enforce = (
        request.node.get_closest_marker("legacy_technology_entities") is None
    )
    import copy

    from lctrend.llm import validation

    original = validation.load_catalog

    def relaxed(name):
        value = original(name)
        if name == "llm_schema":
            value = copy.deepcopy(value)
            value["technology_contract"]["enforce"] = enforce
        return value

    monkeypatch.setattr(validation, "load_catalog", relaxed)


@pytest.fixture(autouse=True)
def technology_triage_mode(request, monkeypatch):
    """The triage call (llm.pipeline._triage) adds one model answer per
    document; tests with recorded answers predate it. Only tests marked
    ``technology_triage`` run it.
    """
    enabled = request.node.get_closest_marker("technology_triage")
    monkeypatch.setenv("LCTREND_TECHNOLOGY_TRIAGE", "1" if enabled else "0")
