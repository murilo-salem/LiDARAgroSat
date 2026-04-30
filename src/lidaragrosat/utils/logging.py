from __future__ import annotations

import logging
from pathlib import Path

from .io import ensure_dir


def configure_logging(log_dir: str | Path | None = None, experiment_name: str = "run") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_path = ensure_dir(log_dir) / f"{experiment_name}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

