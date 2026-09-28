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
