"""
Centralized logging setup for the Spotify Track Performance Intelligence pipeline.

Every module in src/ imports get_logger(__name__) rather than configuring
logging independently, so log format and destination stay consistent across
ingestion, validation, training, and prediction.
"""

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_DIR = Path("reports") / "logs"
_LOG_FILE = _LOG_DIR / "pipeline.log"

_configured = False


def _configure_root_logger(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-scoped logger with console + file handlers attached."""
    _configure_root_logger(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
