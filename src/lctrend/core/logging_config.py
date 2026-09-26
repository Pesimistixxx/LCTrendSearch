"""Project-wide logging: short messages on stderr, full detail in a file.

Console output stays readable while the rotating log file keeps DEBUG
records with timestamps and module names for later diagnosis. Stdout is
left for command results (e.g. parsed documents as JSON).
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Optional, Union

DEFAULT_LOG_FILE = Path("logs") / "lctrend.log"
CONSOLE_FORMAT = "%(levelname)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(process)d] %(message)s"
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5

_HANDLER_MARK = "_lctrend_handler"


class _ConsoleFormatter(logging.Formatter):
    """One line per record; tracebacks are kept for the log file."""

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        return self.formatMessage(record)


def _level(value: Union[str, int, None], default: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return value
    level = logging.getLevelName(str(value).strip().upper())
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level: {value!r}")
    return level


def setup_logging(
    console_level: Union[str, int, None] = None,
    log_file: Optional[Path] = None,
    file_level: Union[str, int, None] = None,
    loggers: Iterable[str] = ("lctrend",),
) -> Path:
    """Attach console and file handlers to the project loggers.

    Repeated calls replace the handlers instead of duplicating records.
    Environment overrides: LCTREND_LOG_LEVEL (console), LCTREND_LOG_FILE,
    LCTREND_LOG_FILE_LEVEL. Returns the path of the log file.
    """
    console = _level(
        console_level or os.getenv("LCTREND_LOG_LEVEL"), logging.INFO
    )
    detail = _level(
        file_level or os.getenv("LCTREND_LOG_FILE_LEVEL"), logging.DEBUG
    )
    path = Path(
        log_file or os.getenv("LCTREND_LOG_FILE") or DEFAULT_LOG_FILE
    ).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(console)
    stream_handler.setFormatter(_ConsoleFormatter(CONSOLE_FORMAT))

    file_handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(detail)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    for handler in (stream_handler, file_handler):
        setattr(handler, _HANDLER_MARK, True)

    for name in loggers:
        logger = logging.getLogger(name)
        remove_handlers(logger)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(min(console, detail))
        # Records stay in the project handlers; the root logger may belong
        # to an embedding application.
        logger.propagate = False
    return path


def remove_handlers(logger: logging.Logger) -> None:
    """Detach and close handlers installed by :func:`setup_logging`."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()
