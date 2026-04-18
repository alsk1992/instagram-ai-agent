"""Centralized logging — rich console + file, JSON-ish for grep."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from rich.logging import RichHandler

_configured = False


def setup_logging(name: str = "ig-agent", logfile: Path | None = None) -> logging.Logger:
    global _configured
    logger = logging.getLogger(name)
    if _configured:
        return logger

    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=True,
        show_path=False,
        markup=True,
    )
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(fh)

    _configured = True

    # Quiet down noisy third-party loggers
    for noisy in ("httpx", "urllib3", "apscheduler", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sys.excepthook = lambda et, ev, tb: logger.exception("Uncaught exception", exc_info=(et, ev, tb))
    return logger


def get_logger(name: str = "ig-agent") -> logging.Logger:
    return logging.getLogger(name)
