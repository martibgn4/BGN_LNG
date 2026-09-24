"""YAML config access. No numeric constant lives in a .py file in this package.

`tests/test_no_magic_numbers.py` parses every module's AST and fails on a float
literal outside {0.0, 1.0, -1.0}, the same rule the sibling GasImbalanceModel
enforces. Anything tunable therefore has to arrive through here.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_FILES = ("target", "sources", "features", "regas", "model")


class ConfigError(RuntimeError):
    """A config file is missing, malformed, or missing a required key."""


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load one config file by stem, e.g. ``load("model")``."""
    if name not in _FILES:
        raise ConfigError(f"unknown config {name!r}; expected one of {_FILES}")
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"missing config file {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get(name: str, *keys: str) -> Any:
    """Fetch a nested key, raising with the full path rather than KeyError.

    A missing setting is a bug in the config, not something to default around,
    so this never takes a default value.
    """
    node = load(name)
    trail: list[str] = []
    for key in keys:
        trail.append(key)
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"{name}.yaml is missing {'.'.join(trail)}")
        node = node[key]
    return node


def config_hash() -> str:
    """Stable hash of all config, stamped onto every run for auditability."""
    blob = json.dumps({f: load(f) for f in _FILES}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
