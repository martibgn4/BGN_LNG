"""Bloomberg adapter tests: config merge, mock, local file, live refusal. No vendor calls."""

from __future__ import annotations

import importlib.util
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    CURVE_SCHEMA,
    OBSERVATION_SCHEMA,
    Backend,
    BackendUnavailable,
    DataError,
    IdentifierUnknown,
    MarketDataAdapter,
    validate_frame,
)
from ttf_forecaster.contracts.futures import ExpiryMismatchError
from ttf_forecaster.data.bloomberg import (
    BloombergConfig,
    LiveBloombergAdapter,
    LocalFileBloombergAdapter,
    MockBloombergAdapter,
    make_market_adapter,
)
from ttf_forecaster.settings import CONFIG_DIR, ConfigError, load_config


class _ExplodingClient:
    """Stands in for xbbg.blp: any call fails the test (proves no request was attempted)."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"live client called: {name}")


@pytest.fixture()
def cfg_dir(tmp_path: Path) -> Path:
    """Copy of configs/ so tests can add a local override without touching the repo."""
    dst = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, dst, ignore=shutil.ignore_patterns("local_*"))
    return dst


def _authorise(cfg_dir: Path) -> None:
    (cfg_dir / "local_data_sources.yaml").write_text("live_calls_authorised: true\n",
                                                     encoding="utf-8")
    load_config.cache_clear()


class TestConfig:
    def test_serves_every_bloomberg_file(self) -> None:
        cfg = BloombergConfig.load()
        assert "bloomberg.yaml" in cfg.files
        for path in sorted((CONFIG_DIR / "sources").glob("bloomberg*.yaml")):
            series = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("series") or {}
            assert set(series) <= set(cfg.series), path.name
        assert cfg.series["fx.eurusd.spot_usd_per_eur"].is_known
        # D-034: DEDAILY has a ticker but no data on this Terminal -> stays not-known.
        assert not cfg.series["price.power_de.daily_eur_mwh"].is_known
        assert cfg.series["price.ttf.da_eur_mwh"].is_known
        assert cfg.live_calls_authorised is False

    def test_duplicate_series_rejected(self, cfg_dir: Path) -> None:
        dup = {"series": {"fx.eurusd.spot_usd_per_eur": {
            "unit": "USD/EUR", "status": "known", "pit_grade": "A", "ticker": "X", "field": "y",
            "publication_lag": {"days": 0}}}}
        (cfg_dir / "sources" / "bloomberg_zz_test.yaml").write_text(yaml.safe_dump(dup),
                                                                    encoding="utf-8")
        with pytest.raises(ConfigError, match="defined in both"):
            BloombergConfig.load(str(cfg_dir))

    def test_unknown_series_and_field(self) -> None:
        cfg = BloombergConfig.load()
        with pytest.raises(IdentifierUnknown):
            cfg.spec("not.a.series")
        with pytest.raises(IdentifierUnknown):
            cfg.field_name("bid")
        assert cfg.field_name("settle") == "px_settle"


class TestMock:
    def test_protocol_and_curve(self) -> None:
        a = make_market_adapter("mock")
        assert isinstance(a, MarketDataAdapter) and a.backend == Backend.MOCK
        curve = a.get_curve_snapshot("TTF", date(2023, 1, 2), date(2023, 3, 31))
        validate_frame(curve, CURVE_SCHEMA)
        assert (curve["source"] == "SYNTHETIC").all()
        pub_local = curve["publication_time"].dt.tz_convert("Europe/Amsterdam")
        assert (pub_local.dt.hour == 19).all()

    def test_series_publication_lag_from_config(self) -> None:
        a = make_market_adapter("mock")
        obs = a.get_historical_series(["fx.eurusd.spot_usd_per_eur"], date(2023, 1, 2),
                                      date(2023, 1, 31))
        validate_frame(obs, OBSERVATION_SCHEMA)
        by_event = obs.set_index("event_time")
        thu = by_event.loc[pd.Timestamp("2023-01-04 23:00", tz="UTC")]   # local Thu 2023-01-05
        fri = by_event.loc[pd.Timestamp("2023-01-05 23:00", tz="UTC")]   # local Fri 2023-01-06
        # next business day 19:00 Amsterdam (18:00 UTC in winter)
        assert thu.publication_time == pd.Timestamp("2023-01-06 18:00", tz="UTC")
        assert fri.publication_time == pd.Timestamp("2023-01-09 18:00", tz="UTC")
        assert bool(fri.publication_time_estimated) and fri.pit_grade == "A"

    def test_metadata(self) -> None:
        meta = make_market_adapter("mock").get_contract_metadata(["TTF.M.2025-11"])
        validate_frame(meta, CONTRACT_METADATA_SCHEMA)
        local = meta["last_trade_date"].dt.tz_convert("Europe/Amsterdam").iloc[0]
        assert local.date() == date(2025, 10, 30)


class TestLocalFile:
    def test_round_trip_and_expiry_check(self, tmp_path: Path) -> None:
        (tmp_path / "curves").mkdir()
        (tmp_path / "contracts").mkdir()
        (tmp_path / "observations").mkdir()
        pd.DataFrame({"trade_date": ["2025-10-01", "2025-10-02"],
                      "ticker": ["TZTX25 Comdty", "TZTX25 Comdty"],
                      "value": [31.0, 32.0]}).to_csv(tmp_path / "curves" / "TTF.csv", index=False)
        pd.DataFrame({"event_time": ["2025-10-01", "2025-10-02"], "value": [1.17, 1.18]}).to_csv(
            tmp_path / "observations" / "fx.eurusd.spot_usd_per_eur.csv", index=False)
        pd.DataFrame({"contract_id": ["TTF.M.2025-11"], "last_trade_date": ["2025-10-30"]}).to_csv(
            tmp_path / "contracts" / "TTF.csv", index=False)
        a = LocalFileBloombergAdapter(extracts_dir=tmp_path)
        curve = a.get_curve_snapshot("TTF", date(2025, 10, 1), date(2025, 10, 31))
        assert list(curve["contract_id"].unique()) == ["TTF.M.2025-11"]
        assert (curve["source"] == "bloomberg").all() and len(curve) == 2
        obs = a.get_historical_series(["fx.eurusd.spot_usd_per_eur"], date(2025, 10, 1),
                                      date(2025, 10, 1))
        assert len(obs) == 1
        meta = a.get_contract_metadata(["TTF.M.2025-11"])
        assert meta["last_trade_date_source"].iloc[0] == "vendor"
        pd.DataFrame({"contract_id": ["TTF.M.2025-11"], "last_trade_date": ["2025-10-29"]}).to_csv(
            tmp_path / "contracts" / "TTF.csv", index=False)
        # D-035 (shipped config, vendor_authoritative: true): the vendor date wins and the rule
        # disagreement is REPORTED, not dropped.
        meta = a.get_contract_metadata(["TTF.M.2025-11"])
        assert str(meta["last_trade_date"].dt.tz_convert("Europe/Amsterdam").dt.date.iloc[0]) \
            == "2025-10-29"
        assert meta.attrs["expiry_mismatches"] == [
            {"contract_id": "TTF.M.2025-11", "rule": "2025-10-30", "vendor": "2025-10-29"}]
        # Strict mode (vendor_authoritative: false) keeps the hard error.
        strict = tmp_path / "configs"
        shutil.copytree(CONFIG_DIR, strict, ignore=shutil.ignore_patterns("local_*"))
        inst = strict / "instruments.yaml"
        inst.write_text(inst.read_text(encoding="utf-8").replace(
            "vendor_authoritative: true", "vendor_authoritative: false"), encoding="utf-8")
        load_config.cache_clear()
        b = LocalFileBloombergAdapter(extracts_dir=tmp_path, config_dir=str(strict))
        with pytest.raises(ExpiryMismatchError):
            b.get_contract_metadata(["TTF.M.2025-11"])


class _CurveClient:
    """Fake xbbg.blp.bdh: data only under the tickers in ``live`` (one value per day)."""

    def __init__(self, live: dict[str, float]) -> None:
        self.live = live
        self.calls = 0

    def bdh(self, tickers: list[str], fields: list[str], start: date, end: date) -> pd.DataFrame:
        self.calls += 1
        idx = pd.date_range(start, end, freq="B")
        cols, data = [], {}
        for t in tickers:
            if t in self.live:
                for f in fields:
                    cols.append((t, f.lower()))
                    data[(t, f.lower())] = [self.live[t]] * len(idx)
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(data, index=idx, columns=pd.MultiIndex.from_tuples(cols))


class TestLivePnxResolution:
    """D-039: a pnx hub (EEX/PNX root) asks for both the listed and the expired ticker and keeps the one
    that returns data. Both forms returning data for one contract is refused, never guessed."""

    def test_expired_form_resolves_and_is_logged(self, cfg_dir: Path) -> None:
        _authorise(cfg_dir)
        client = _CurveClient({"PNGX6M16 Comdty": 15.9})
        a = LiveBloombergAdapter(config_dir=str(cfg_dir), client=client)
        out = a.get_curve_snapshot("PEG", date(2016, 10, 3), date(2016, 10, 7))
        assert set(out["contract_id"]) == {"PEG.M.2016-11"}
        assert a.last_ticker_counts == {"PNGX6M16 Comdty": 1}

    def test_listed_form_resolves(self, cfg_dir: Path) -> None:
        _authorise(cfg_dir)
        client = _CurveClient({"PNGX6M Comdty": 74.0})
        a = LiveBloombergAdapter(config_dir=str(cfg_dir), client=client)
        out = a.get_curve_snapshot("PEG", date(2026, 10, 1), date(2026, 10, 7))
        assert "PEG.M.2026-11" in set(out["contract_id"])
        assert a.last_ticker_counts.get("PNGX6M Comdty") == 1

    def test_both_forms_identical_data_is_one_security(self, cfg_dir: Path) -> None:
        # the listed and expired forms of one listed contract return the same series: accepted,
        # the expired form (it names the year) is the one counted
        _authorise(cfg_dir)
        client = _CurveClient({"PNGX6M16 Comdty": 15.9, "PNGX6M Comdty": 15.9})
        a = LiveBloombergAdapter(config_dir=str(cfg_dir), client=client)
        out = a.get_curve_snapshot("PEG", date(2016, 10, 3), date(2016, 10, 7))
        assert set(out["contract_id"]) == {"PEG.M.2016-11"}
        assert a.last_ticker_counts == {"PNGX6M16 Comdty": 1}

    def test_both_forms_disagreeing_data_is_refused(self, cfg_dir: Path) -> None:
        _authorise(cfg_dir)
        client = _CurveClient({"PNGX6M16 Comdty": 15.9, "PNGX6M Comdty": 16.4})
        a = LiveBloombergAdapter(config_dir=str(cfg_dir), client=client)
        with pytest.raises(DataError, match="disagree"):
            a.get_curve_snapshot("PEG", date(2016, 10, 3), date(2016, 10, 7))


class TestLive:
    def test_refuses_without_authorisation(self) -> None:
        with pytest.raises(BackendUnavailable, match="not authorised"):
            make_market_adapter("live", client=_ExplodingClient())

    @pytest.mark.skipif(importlib.util.find_spec("xbbg") is None
                        or importlib.util.find_spec("blpapi") is None,
                        reason="xbbg/blpapi not installed: environment check would refuse first")
    def test_refuses_unknown_identifier_before_any_call(self, cfg_dir: Path) -> None:
        _authorise(cfg_dir)
        a = LiveBloombergAdapter(config_dir=str(cfg_dir), client=_ExplodingClient())
        with pytest.raises(IdentifierUnknown, match="price.power_de.daily_eur_mwh"):
            a.get_historical_series(["fx.eurusd.spot_usd_per_eur", "price.power_de.daily_eur_mwh"],
                                    date(2025, 1, 1), date(2025, 1, 31))
        with pytest.raises(IdentifierUnknown):
            a.get_curve_snapshot("TTF", date(2025, 1, 1), date(2025, 1, 31), fields=("bid",))

    def test_unknown_backend(self) -> None:
        with pytest.raises(ConfigError):
            make_market_adapter("ftp")
