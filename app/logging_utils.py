from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .constants import APP_NAME, DEFAULT_LOG_FILE


def setup_logging(log_level: str, log_file: str | Path | None = None) -> logging.Logger:
    """Configure console and rotating file logging."""

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    target_file = Path(log_file or DEFAULT_LOG_FILE)
    target_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        target_file,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
