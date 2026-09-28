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
    yield
    for name in ("lctrend", "frontend.server"):
        logger = logging.getLogger(name)
        remove_handlers(logger)
        logger.propagate = True
