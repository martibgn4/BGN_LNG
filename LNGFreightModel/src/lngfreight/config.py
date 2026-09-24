"""Config access. Every tunable lives in ``config/*.yaml``, never in code.

``get("model", "estimators", "ridge", "alphas")`` reads model.yaml. A missing
key raises rather than returning a default, because a silently defaulted
hyperparameter is a result that cannot be reproduced from the config alone.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROJECT_DIR = CONFIG_DIR.parent


class ConfigError(KeyError):
    """A requested config path does not exist."""


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise ConfigError(f"no config file {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get(name: str, *path: str) -> Any:
    node: Any = load(name)
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"{name}.yaml has no key {'.'.join(walked)}")
        node = node[key]
    return node


def cache_dir() -> Path:
    d = PROJECT_DIR / "data_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir() -> Path:
    d = PROJECT_DIR / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d
