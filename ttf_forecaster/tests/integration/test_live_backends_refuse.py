"""Live backends refuse without authorisation and never touch the network.

The socket layer is patched to raise (``no_network`` fixture), so a refusal that happened only
*after* a connection attempt would fail the test. Configs come from a pristine copy of the committed
``configs/`` (no ``local*.yaml``), so a developer's local authorisation cannot leak in.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ttf_forecaster.data.base import BackendUnavailable, IdentifierUnknown
from ttf_forecaster.data.bloomberg import BloombergConfig, LiveBloombergAdapter, make_market_adapter
from ttf_forecaster.data.fundamentals import (
    LiveFundamentalsAdapter,
    load_fundamentals_registry,
    make_fundamentals_adapter,
)
from ttf_forecaster.data.spark import LiveSparkAdapter, make_spark_adapter
from ttf_forecaster.data.weather import LiveWeatherAdapter, make_weather_adapter
from ttf_forecaster.settings import CONFIG_DIR

LOCAL_DS = CONFIG_DIR / "local_data_sources.yaml"
needs_no_local_override = pytest.mark.skipif(
    LOCAL_DS.exists(), reason="configs/local_data_sources.yaml present; default-path test skipped"
)


# -- Bloomberg ---------------------------------------------------------------------------------


def test_bloomberg_live_refuses(no_network: list[str], clean_config_dir: Path) -> None:
    cfg = BloombergConfig.load(str(clean_config_dir))
    assert cfg.live_calls_authorised is False
    with pytest.raises(BackendUnavailable, match="not authorised"):
        make_market_adapter("live", config_dir=str(clean_config_dir))
    with pytest.raises(BackendUnavailable, match="not authorised"):
        LiveBloombergAdapter(cfg)


def test_bloomberg_live_refuses_even_with_injected_client(no_network: list[str],
                                                          clean_config_dir: Path) -> None:
    """An injected client must not bypass the authorisation check."""
    calls: list[str] = []

    class _Spy:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            calls.append(name)
            raise AssertionError(f"client.{name} called before authorisation")

    with pytest.raises(BackendUnavailable):
        LiveBloombergAdapter(config_dir=str(clean_config_dir), client=_Spy())
    assert calls == []


# -- Spark -------------------------------------------------------------------------------------


class _ExplodingTransport:
    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        raise AssertionError(f"transport.{name} used before authorisation")


def _no_secret(*_a: object, **_k: object) -> str | None:
    raise AssertionError("a secret was read before authorisation")


def test_spark_live_refuses_explicit(no_network: list[str]) -> None:
    with pytest.raises(BackendUnavailable, match="not authorised"):
        LiveSparkAdapter(live_calls_authorised=False, transport=_ExplodingTransport(),  # type: ignore[arg-type]
                         secret_getter=_no_secret)


@needs_no_local_override
def test_spark_live_refuses_by_default(no_network: list[str]) -> None:
    with pytest.raises(BackendUnavailable, match="not authorised"):
        make_spark_adapter("live", transport=_ExplodingTransport(), secret_getter=_no_secret)


# -- Fundamentals ------------------------------------------------------------------------------


def test_fundamentals_live_refuses_known_series(no_network: list[str], clean_config_dir: Path
                                                ) -> None:
    reg = load_fundamentals_registry(str(clean_config_dir))
    known = [k for k, v in reg.items() if v.status.value == "known"]
    assert known, "expected at least one known fundamentals series"
    ad = LiveFundamentalsAdapter(reg, authorised=False)
    with pytest.raises(BackendUnavailable, match="not authorised"):
        ad.get_series(known[:1], date(2024, 1, 1), date(2024, 1, 31))


def test_fundamentals_live_refuses_unknown_series_first(no_network: list[str],
                                                        clean_config_dir: Path) -> None:
    reg = load_fundamentals_registry(str(clean_config_dir))
    unknown = [k for k, v in reg.items() if v.status.value != "known"]
    assert unknown
    ad = LiveFundamentalsAdapter(reg, authorised=True)  # even when authorised
    with pytest.raises(IdentifierUnknown):
        ad.get_series(unknown[:1], date(2024, 1, 1), date(2024, 1, 31))


@needs_no_local_override
def test_fundamentals_live_default_is_unauthorised(no_network: list[str]) -> None:
    ad = make_fundamentals_adapter("live")
    assert ad.authorised is False  # type: ignore[union-attr]


# -- Weather -----------------------------------------------------------------------------------


def test_weather_live_refuses(no_network: list[str], clean_config_dir: Path) -> None:
    with pytest.raises(BackendUnavailable, match="not authorised"):
        make_weather_adapter("live", config_dir=str(clean_config_dir))
    with pytest.raises(BackendUnavailable, match="not authorised"):
        LiveWeatherAdapter(config_dir=str(clean_config_dir))


def test_weather_live_refuses_licence_even_if_authorised(no_network: list[str],
                                                         clean_config_dir: Path) -> None:
    with pytest.raises(BackendUnavailable, match="NON-COMMERCIAL"):
        LiveWeatherAdapter(live_calls_authorised=True, config_dir=str(clean_config_dir))


# -- Defaults ----------------------------------------------------------------------------------


@needs_no_local_override
def test_default_backends_are_mock(no_network: list[str]) -> None:
    assert make_market_adapter().backend.value == "mock"
    assert make_spark_adapter().backend.value == "mock"
    assert make_fundamentals_adapter().backend.value == "mock"
    assert make_weather_adapter().backend.value == "mock"
