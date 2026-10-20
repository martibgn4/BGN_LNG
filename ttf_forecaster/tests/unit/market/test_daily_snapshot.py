"""Daily-snapshot Terminal preflight: a clear message when the Bloomberg Terminal is not connected."""
from __future__ import annotations

import pytest

from ttf_forecaster.pipelines.daily_snapshot import TERMINAL_NOT_CONNECTED, TerminalNotConnected, check_terminal


class FakeSession:
    def __init__(self, start_ok: bool = True, service_ok: bool = True) -> None:
        self.start_ok, self.service_ok = start_ok, service_ok
        self.stopped = False
        self.services: list[str] = []

    def start(self) -> bool:
        return self.start_ok

    def openService(self, name: str) -> bool:  # noqa: N802 - blpapi's name
        self.services.append(name)
        return self.service_ok

    def stop(self) -> None:
        self.stopped = True


def test_connected_terminal_passes_and_closes_session():
    s = FakeSession()
    check_terminal(lambda: s)
    assert s.services == ["//blp/refdata"]
    assert s.stopped


def test_session_not_started_raises_clear_message():
    with pytest.raises(TerminalNotConnected, match="Bloomberg Terminal not connected"):
        check_terminal(lambda: FakeSession(start_ok=False))


def test_refdata_not_opened_raises_and_closes_session():
    s = FakeSession(service_ok=False)
    with pytest.raises(TerminalNotConnected) as err:
        check_terminal(lambda: s)
    assert str(err.value) == TERMINAL_NOT_CONNECTED
    assert s.stopped


def test_script_exits_2_with_the_message_before_any_pull(monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "scripts" / "run_daily_snapshot.py"
    spec = importlib.util.spec_from_file_location("run_daily_snapshot_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def not_connected():
        raise TerminalNotConnected(TERMINAL_NOT_CONNECTED)

    def must_not_run(*_a, **_k):
        raise AssertionError("no config copy, store or pull may happen when the Terminal is down")

    monkeypatch.setattr(mod, "check_terminal", not_connected)
    monkeypatch.setattr(mod, "authorised_config_dir", must_not_run)
    monkeypatch.setattr(mod, "ParquetStore", must_not_run)
    assert mod.main([]) == 2
    assert "Bloomberg Terminal not connected" in capsys.readouterr().err


def test_connection_exception_is_wrapped_with_the_clear_message():
    def boom():
        raise OSError("connection refused")

    with pytest.raises(TerminalNotConnected, match="log in to the Terminal.*OSError: connection refused"):
        check_terminal(boom)
