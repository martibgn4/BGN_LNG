"""Project settings: paths, YAML configuration and secrets.

Configuration is split across ``configs/*.yaml`` (committed, identifiers and parameters only) and
``configs/local*.yaml`` / ``.env`` (git-ignored, machine-specific). Secrets are read only from the
process environment or ``.env`` and are never logged or returned in a repr.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_DIR: Path = PROJECT_ROOT / "configs"
DATA_LAYERS: tuple[str, ...] = ("raw", "normalised", "features", "predictions", "benchmarks")


class ConfigError(RuntimeError):
    """A configuration file is missing, malformed, or lacks a required key."""


class SecretMissingError(RuntimeError):
    """A required secret is not set. The message names the variable, never a value."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at top level")
    return content


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (mappings merge, other values replace)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_deep_merge = deep_merge  # backwards-compatible private alias


@lru_cache(maxsize=None)
def load_config(name: str, config_dir: str | None = None) -> dict[str, Any]:
    """Load ``configs/<name>.yaml`` merged with an optional git-ignored ``configs/local_<name>.yaml``.

    Args:
        name: config stem, e.g. ``"targets"`` or ``"sources/bloomberg"``.
        config_dir: override directory (tests).

    Returns:
        The merged mapping. Cached; call ``load_config.cache_clear()`` after editing files.
    """
    base_dir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg = _read_yaml(base_dir / f"{name}.yaml")
    local = base_dir / f"local_{Path(name).name}.yaml"
    if local.exists():
        cfg = deep_merge(cfg, _read_yaml(local))
    return cfg


def get(cfg: Mapping[str, Any], *keys: str) -> Any:
    """Fetch a nested key, raising ``ConfigError`` naming the full path if absent."""
    node: Any = cfg
    for i, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node:
            raise ConfigError(f"Missing config key: {'.'.join(keys[: i + 1])}")
        node = node[key]
    return node


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal ``.env`` parser (KEY=VALUE, ``#`` comments). Avoids a python-dotenv dependency."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_secret(name: str, *, required: bool = True) -> str | None:
    """Return a secret from the environment, falling back to the project ``.env``.

    Never logs the value. Empty strings count as unset.

    Raises:
        SecretMissingError: if ``required`` and the variable is unset or empty.
    """
    value = os.environ.get(name) or _load_dotenv(PROJECT_ROOT / ".env").get(name)
    if not value:
        if required:
            raise SecretMissingError(
                f"Environment variable {name} is not set. Add it to your environment or to "
                f"{PROJECT_ROOT / '.env'} (git-ignored); see .env.example."
            )
        return None
    return value


@dataclass(frozen=True)
class Paths:
    """Resolved data-layer directories. ``TTF_DATA_ROOT`` overrides the default ``./data``."""

    root: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> Paths:
        chosen = root or get_secret("TTF_DATA_ROOT", required=False) or PROJECT_ROOT / "data"
        return cls(Path(chosen))

    def layer(self, name: str) -> Path:
        if name not in DATA_LAYERS:
            raise ConfigError(f"Unknown data layer {name!r}; expected one of {DATA_LAYERS}")
        return self.root / name

    def ensure(self) -> Paths:
        for layer in DATA_LAYERS:
            self.layer(layer).mkdir(parents=True, exist_ok=True)
        return self
