r"""No secret values or credential paths are read by default.

Two checks:

1. **Static (grep-style):** no committed source, config or test file outside this module refers to
   ``adhoc_scripts/client_credentials.csv`` (tracked in the parent repo, see
   docs/environment_audit.md), any ``*credentials*`` file, or an absolute ``C:\Users`` /
   ``C:\Marti`` path.
2. **Runtime:** the default offline workflows (``check-config``, ``identifiers``, every mock
   adapter) run with ``get_secret`` / ``.env`` loading patched to raise and with every file open
   recorded; nothing named ``.env`` or ``*credential*`` is opened.
"""

from __future__ import annotations

import builtins
import io
import re
from pathlib import Path

import pytest

from ttf_forecaster import cli, settings
from ttf_forecaster.settings import PROJECT_ROOT

THIS_FILE = Path(__file__).resolve()
SCANNED = ("src", "configs", "tests", "scripts", "notebooks")
SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".ipynb", ".cfg", ".ini", ".json", ".md"}
# (pattern, scope) - scope "all" = every scanned file, "src" = package code only. Configs may cite
# adhoc_scripts/*.py as identifier *provenance*; package code must never read from there.
FORBIDDEN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"client_credentials", re.IGNORECASE), "all"),
    (re.compile(r"adhoc_scripts", re.IGNORECASE), "src"),
    (re.compile(r"\b[A-Za-z]:[\\/]+(Users|Marti)\b", re.IGNORECASE), "all"),
]
# Lines that mention a pattern only to state or assert its absence are allowed.
ALLOWED_LINE = re.compile(r"assert .*not in|never read", re.IGNORECASE)


def _scanned_files() -> list[Path]:
    out = []
    for top in SCANNED:
        root = PROJECT_ROOT / top
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if (p.is_file() and p.suffix in SUFFIXES and "__pycache__" not in p.parts
                    and p.resolve() != THIS_FILE and not p.name.startswith("local")):
                out.append(p)
    return out


def test_no_credential_or_absolute_paths_in_project_files() -> None:
    hits = []
    files = _scanned_files()
    assert len(files) > 50, "scan found suspiciously few files"
    for p in files:
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if ALLOWED_LINE.search(line):
                continue
            in_src = p.is_relative_to(PROJECT_ROOT / "src")
            for pat, scope in FORBIDDEN:
                if (scope == "all" or in_src) and pat.search(line):
                    hits.append(f"{p.relative_to(PROJECT_ROOT)}:{i}: {line.strip()[:100]}")
    assert not hits, "forbidden references:\n" + "\n".join(hits)


def test_gitignore_covers_secrets_and_data() -> None:
    gi = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in ("/data/", ".env", "*credentials*", "configs/local*.yaml"):
        assert pattern in gi, f".gitignore lacks {pattern}"


@pytest.fixture
def secrets_forbidden(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch every secret entry point to raise and record every opened file path."""
    opened: list[str] = []

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError(f"secret read by default: {a!r}")

    monkeypatch.setattr(settings, "get_secret", _boom)
    monkeypatch.setattr(settings, "_load_dotenv", _boom)
    import ttf_forecaster.data.spark as spark_mod
    if hasattr(spark_mod, "get_secret"):
        monkeypatch.setattr(spark_mod, "get_secret", _boom)

    real_open = builtins.open

    def _spy(file, *a, **k):  # type: ignore[no-untyped-def]
        opened.append(str(file))
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", _spy)
    monkeypatch.setattr(io, "open", _spy)
    return opened


def test_default_workflows_read_no_secrets(secrets_forbidden: list[str], no_network: list[str],
                                           capsys: pytest.CaptureFixture[str]) -> None:
    from ttf_forecaster.data.bloomberg import make_market_adapter
    from ttf_forecaster.data.fundamentals import make_fundamentals_adapter
    from ttf_forecaster.data.spark import make_spark_adapter
    from ttf_forecaster.data.weather import make_weather_adapter

    assert cli.main(["check-config"]) == 0
    assert cli.main(["identifiers"]) == 0
    for factory in (make_market_adapter, make_spark_adapter, make_fundamentals_adapter,
                    make_weather_adapter):
        factory("mock")
    capsys.readouterr()
    bad = [f for f in secrets_forbidden
           if Path(f).name == ".env" or "credential" in Path(f).name.lower()]
    assert not bad, bad
