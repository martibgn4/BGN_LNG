"""CLI smoke tests: the read-only commands succeed on the committed configs and stay offline."""

from __future__ import annotations

import re

import pytest

from ttf_forecaster import cli


def test_check_config_returns_zero(no_network: list[str], capsys: pytest.CaptureFixture[str]
                                   ) -> None:
    assert cli.main(["check-config"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "adapter registries load (mock)" in out


def test_identifiers_returns_zero(no_network: list[str], capsys: pytest.CaptureFixture[str]
                                  ) -> None:
    assert cli.main(["identifiers"]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1].startswith("TOTAL")
    # every status printed is one of the three allowed values (CLAUDE.md "No invented identifiers")
    allowed = {"known", "requires_discovery", "requires_user_input"}
    statuses = set(re.findall(r"([a-z_]+)=\d+", out))
    assert statuses, out
    assert statuses <= allowed, statuses - allowed

def test_unknown_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])
