"""Configuration loading for reproducible ZSC-TSD experiments.

This module is deliberately lightweight.  Model-specific environments import
the same YAML profiles, while heavyweight libraries remain isolated in their
own Conda environments.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigurationError(ValueError):
    """Raised when an experiment profile is malformed or ambiguous."""


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _merge(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _profile_path(value: str | Path, config_dir: Path) -> Path:
    candidate = Path(value)
    if candidate.suffix:
        return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    return config_dir / f"{candidate}.yaml"


def load_profile(
    profile: str | Path,
    *,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load a YAML profile and recursively merge ``experiment.inherit``.

    ``${NAME}`` environment placeholders are expanded before YAML parsing.
    Missing variables are left unchanged so that a profile can be inspected
    without exposing local credentials.
    """

    root = Path(config_dir).resolve()
    environment = dict(os.environ if environ is None else environ)
    active: list[Path] = []

    def visit(value: str | Path) -> dict[str, Any]:
        path = _profile_path(value, root).resolve()
        if path in active:
            chain = " -> ".join(str(item) for item in (*active, path))
            raise ConfigurationError(f"cyclic configuration inheritance: {chain}")
        if not path.is_file():
            raise FileNotFoundError(f"configuration profile not found: {path}")

        text = path.read_text(encoding="utf-8")
        for name, replacement in environment.items():
            text = text.replace("${" + name + "}", replacement)
        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ConfigurationError(f"configuration root must be an object: {path}")

        active.append(path)
        try:
            experiment = parsed.get("experiment") or {}
            inherits = experiment.get("inherit") or []
            if isinstance(inherits, str):
                inherits = [inherits]
            if not isinstance(inherits, list) or not all(
                isinstance(item, str) and item for item in inherits
            ):
                raise ConfigurationError(
                    f"experiment.inherit must be a string list: {path}"
                )
            combined: dict[str, Any] = {}
            for parent in inherits:
                combined = _merge(combined, visit(parent))
            combined = _merge(combined, parsed)
            combined.setdefault("_meta", {})
            combined["_meta"]["profile_path"] = str(path)
            return combined
        finally:
            active.pop()

    return visit(profile)


def require_mapping(config: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    """Return a required nested mapping with a useful error message."""

    current: Any = config
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise ConfigurationError("missing configuration: " + ".".join(traversed))
        current = current[key]
    if not isinstance(current, Mapping):
        raise ConfigurationError("configuration must be an object: " + ".".join(keys))
    return current
