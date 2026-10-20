"""Integration fixtures (owner: software-quality-agent). SYNTHETIC data only, no network.

Markers registered here so the suite stays runnable without editing ``pyproject.toml``:

* ``integration`` - every test in this directory;
* ``slow``        - the synthetic end-to-end runs (~45 s each). Skip with ``-m "not slow"`` or by
  setting ``TTF_SKIP_SLOW=1``.
"""

from __future__ import annotations

import os
import shutil
import socket
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from ttf_forecaster.settings import CONFIG_DIR, load_config

#: Reduced synthetic window for the e2e tests: 2 gas years of warm-up, 3 years of origins, which
#: yields 4 outer folds under configs/backtests.yaml (min_train_origins=250, 6-month blocks).
#: NOT shorter: with end 2022-12-30 (2 folds) the null world's ridge "passed" criterion 1
#: (fold share 2/2, HLN-DM p=0.004) - a small-sample false positive, see
#: docs/software_quality_review.md SQ-03.
SHORT_SYNTHETIC_END = "2023-12-29"
SHORT_FIRST_ORIGIN = "2021-01-01"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: cross-module integration test (synthetic)")
    config.addinivalue_line("markers", "slow: synthetic end-to-end run; skip with -m 'not slow'")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    skip_slow = pytest.mark.skip(reason="TTF_SKIP_SLOW=1")
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)
            if os.environ.get("TTF_SKIP_SLOW") == "1" and "slow" in item.keywords:
                item.add_marker(skip_slow)


class NetworkTouched(AssertionError):
    """Raised by the socket guard: a test opened (or tried to resolve) a network connection."""


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Make every socket entry point raise; yields the list of attempted calls."""
    attempts: list[str] = []

    def _deny(name: str):  # type: ignore[no-untyped-def]
        def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
            attempts.append(f"{name}{args[:2]!r}")
            raise NetworkTouched(f"network access attempted: {name}")
        return _raise

    monkeypatch.setattr(socket.socket, "connect", _deny("socket.connect"))
    monkeypatch.setattr(socket.socket, "connect_ex", _deny("socket.connect_ex"))
    monkeypatch.setattr(socket, "create_connection", _deny("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", _deny("getaddrinfo"))
    yield attempts
    assert not attempts, f"network was touched: {attempts}"


def copy_configs(dest: Path) -> Path:
    """Copy ``configs/`` to ``dest/configs`` WITHOUT any git-ignored ``local*.yaml`` overrides, so a
    developer's local authorisation never leaks into a test."""
    out = dest / "configs"
    shutil.copytree(CONFIG_DIR, out, ignore=shutil.ignore_patterns("local*", "__pycache__"))
    return out


@pytest.fixture(scope="session")
def clean_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Pristine copy of the committed configs (``live_calls_authorised: false``)."""
    out = copy_configs(tmp_path_factory.mktemp("cfg_clean"))
    ds = yaml.safe_load((out / "data_sources.yaml").read_text(encoding="utf-8"))
    assert ds["live_calls_authorised"] is False, "committed data_sources.yaml must not authorise"
    assert ds["default_backend"] == "mock"
    return out


@pytest.fixture(scope="session")
def short_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Config copy whose ``pipeline.yaml`` has a shorter synthetic window (configs/ is untouched)."""
    out = copy_configs(tmp_path_factory.mktemp("cfg_short"))
    p = yaml.safe_load((out / "pipeline.yaml").read_text(encoding="utf-8"))
    p["synthetic"]["end"] = SHORT_SYNTHETIC_END
    p["evaluation"]["first_origin"] = SHORT_FIRST_ORIGIN
    (out / "pipeline.yaml").write_text(yaml.safe_dump(p, sort_keys=False), encoding="utf-8")
    load_config.cache_clear()
    return out
