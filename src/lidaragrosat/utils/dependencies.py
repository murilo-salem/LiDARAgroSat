from __future__ import annotations

import importlib


class MissingOptionalDependencyError(ImportError):
    """Raised when an optional dependency is required by a code path."""


def require_dependency(module_name: str, purpose: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise MissingOptionalDependencyError(
            f"Dependency '{module_name}' is required for {purpose}. "
            f"Install the project dependencies defined in pyproject.toml."
        ) from exc

