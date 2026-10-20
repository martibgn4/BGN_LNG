"""SYNTHETIC bundles for feature tests. Nothing here is market data."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    MARKET_TZ,
    DataBundle,
    empty_frame,
)
from ttf_forecaster.data.synthetic import (
    SyntheticConfig,
    storage_observations,
    ttf_curve,
    weather_forecast_runs,
)

CFG = SyntheticConfig(start="2022-01-03", end="2023-06-30")


def contract_meta(curve: pd.DataFrame, bdays: int = 2) -> pd.DataFrame:
    """Metadata consistent with the synthetic expiry rule (delivery start - 2 business days)."""
    c = curve.drop_duplicates("contract_id")[["contract_id", "hub", "tenor_type", "delivery_start",
                                               "delivery_end", "currency"]].copy()
    local = c["delivery_start"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    ltd = (local - pd.offsets.BDay(bdays)).dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
    c["last_trade_date"] = ltd
    c["last_trade_date_source"] = "rule"
    c["unit"] = "EUR/MWh"
    out = empty_frame(CONTRACT_METADATA_SCHEMA)
    return pd.concat([out, c[out.columns]], ignore_index=True)


def cross_curve(ttf: pd.DataFrame, hub: str, unit: str, transform: Any, pub_days: int,
                tz: str) -> pd.DataFrame:
    """Another hub's curve derived from TTF rows (same trade dates / delivery months)."""
    df = ttf.copy()
    df["hub"] = hub
    df["contract_id"] = df["contract_id"].str.replace("TTF", hub)
    df["value"] = transform(df["value"].to_numpy())
    df["unit"] = unit
    df["currency"] = unit.split("/")[0][:3].upper()
    # re-express trade_date as local midnight in the hub's own timezone
    local_day = df["trade_date"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    df["trade_date"] = local_day.dt.tz_localize(tz).dt.tz_convert("UTC")
    df["source_timezone"] = tz
    pub_local = (local_day + pd.offsets.BDay(pub_days) + pd.Timedelta(hours=19))
    df["publication_time"] = pub_local.dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
    df["ingestion_time"] = df["publication_time"] + pd.Timedelta(minutes=5)
    return df


def fx_obs(days: pd.DatetimeIndex, series_id: str, level: float, seed: int) -> pd.DataFrame:
    """Daily FX (units per EUR), next-business-day 19:00 publication (PIT §4)."""
    rng = np.random.default_rng(seed)
    n = len(days)
    pub = (days + pd.offsets.BDay(1) + pd.Timedelta(hours=19)).tz_localize(MARKET_TZ).tz_convert(
        "UTC")
    return pd.DataFrame({
        "series_id": series_id,
        "event_time": days.tz_localize(MARKET_TZ).tz_convert("UTC"),
        "value": level * np.exp(np.cumsum(rng.normal(0, 0.003, n))),
        "geography": "GLOBAL",
        "publication_time": pub,
        "ingestion_time": pub + pd.Timedelta(minutes=5),
        "revision": np.zeros(n, dtype="int64"),
        "unit": "ccy/EUR",
        "source": "SYNTHETIC",
        "quality_flag": "synthetic",
        "entitlement": "synthetic",
        "pit_grade": "A",
        "publication_time_estimated": np.zeros(n, dtype=bool),
        "source_timezone": MARKET_TZ,
    })


def bundle(cfg: SyntheticConfig = CFG, *, with_cross: bool = True) -> DataBundle:
    ttf = ttf_curve(cfg)
    curves = [ttf]
    obs = [storage_observations(cfg)]
    if with_cross:
        curves += [
            cross_curve(ttf, "JKM", "USD/MMBtu", lambda v: v * 0.35 + 1.0, 1, "Asia/Tokyo"),
            cross_curve(ttf, "HH", "USD/MMBtu", lambda v: v * 0.1 + 2.0, 1, "America/New_York"),
            cross_curve(ttf, "Brent", "USD/bbl", lambda v: v * 0.0 + 80.0, 1, "Europe/London"),
            cross_curve(ttf, "NBP", "GBp/therm", lambda v: v * 7.0, 0, "Europe/London"),
            cross_curve(ttf, "THE", "EUR/MWh", lambda v: v + 1.5, 0, "Europe/Berlin"),
        ]
        days = pd.bdate_range(cfg.start, cfg.end)
        obs += [fx_obs(days, "fx.eurusd.spot_usd_per_eur", 1.08, 1),
                fx_obs(days, "fx.eurgbp.spot_gbp_per_eur", 0.86, 2)]
    return DataBundle(
        observations=pd.concat(obs, ignore_index=True),
        forecasts=weather_forecast_runs(cfg),
        curves=pd.concat(curves, ignore_index=True),
        contracts=contract_meta(ttf),
    )


def settle_origins(start: str, end: str) -> pd.DatetimeIndex:
    """Origins at 19:00 Europe/Amsterdam on business days (after TTF settlement publication)."""
    d = pd.bdate_range(start, end)
    return (d + pd.Timedelta(hours=19)).tz_localize(MARKET_TZ).tz_convert("UTC")


def config() -> dict[str, Any]:
    from ttf_forecaster.pipelines.build_features import load_feature_config

    return copy.deepcopy(load_feature_config())
