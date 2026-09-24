"""Config loading. SPEC 3: every assumption lives in YAML, none in Python.

Also computes the config hash that SPEC 9 requires on every output.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ConfigError(RuntimeError):
    """Raised when config is missing, malformed, or internally inconsistent."""


def config_path(name: str) -> Path:
    p = CONFIG_DIR / name
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    return p


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load a YAML config file. Cached - config is immutable within a run."""
    with config_path(name).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"{name} did not parse to a mapping")
    return data


def config_hash() -> str:
    """Stable hash over every YAML file under config/, for output provenance.

    Includes generated/ so that a re-resolved point registry produces a
    different hash - SPEC 9 wants outputs traceable to the exact inputs.
    """
    digest = hashlib.sha256()
    for path in sorted(CONFIG_DIR.rglob("*.yaml")):
        digest.update(path.relative_to(CONFIG_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]
