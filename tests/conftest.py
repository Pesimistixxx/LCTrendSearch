import logging

import pytest

from lctrend.core.logging_config import remove_handlers


@pytest.fixture(autouse=True)
def isolated_log_file(monkeypatch, tmp_path):
    """Keep CLI runs from writing to the project's logs/ directory."""
    monkeypatch.setenv("LCTREND_LOG_FILE", str(tmp_path / "lctrend.log"))
    yield
    for name in ("lctrend", "frontend.server"):
        logger = logging.getLogger(name)
        remove_handlers(logger)
        logger.propagate = True
