from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_text(payload: str, path: str | Path) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(payload, encoding="utf-8")

