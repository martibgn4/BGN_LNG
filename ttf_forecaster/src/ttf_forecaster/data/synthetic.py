"""Synthetic data generators. **Every value here is fake** and labelled ``SYNTHETIC``.

Two worlds, so the pipeline can be tested in both directions:

* ``world="null"`` — prices are a random walk **independent** of every fundamental. Any model
  that shows out-of-sample skill here has a leak. This is the default.
* ``world="planted"`` — a known, documented fraction of next-day return is driven by the lagged
  storage surprise. A correct pipeline must find it; a pipeline that cannot is broken.

Nothing produced here may be quoted as evidence about TTF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    MARKET_TZ,
    SYNTHETIC_SOURCE,
    Entitlement,
    PitGrade,
    QualityFlag,
    contract_id,
)

SETTLE_HOUR_LOCAL = 19  # matches target_definitions.md publication convention


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic world. Defaults give ~4 years of daily data."""

    start: str = "2021-01-04"
    end: str = "2024-12-31"
    seed: int = 7
    world: str = "null"  # "null" or "planted"
    planted_beta: float = 0.004  # daily log-return per unit storage surprise (planted world)
    daily_vol: float = 0.03
    months_listed: int = 18
    expiry_bdays_before_delivery: int = 2
    storage_pub_lag_days: int = 2
    storage_revision_prob: float = 0.2
    contract_noise: float = 0.0  # iid daily log-noise per contract (0 = single common factor)
    planted_series_id: str = "storage.eu.inventory_twh"
    # "level": storage deviation from its 30d mean (persistent -> also visible in past returns);
    # "change": day-on-day inventory change vs its 30d mean (near-white flow surprise -> only a
    # storage feature can carry it, so it tests whether fundamentals beat price-only models).
    planted_driver: str = "level"


def trading_days(cfg: SyntheticConfig) -> pd.DatetimeIndex:
    """Weekday calendar (no holidays) as local dates — synthetic, not the ICE Endex calendar."""
    return pd.bdate_range(cfg.start, cfg.end)


def _settle_ts(local_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (
        (local_dates + pd.Timedelta(hours=SETTLE_HOUR_LOCAL))
        .tz_localize(MARKET_TZ)
        .tz_convert("UTC")
    )


def _midnight_utc(local_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return local_dates.tz_localize(MARKET_TZ).tz_convert("UTC")


def _provenance(n: int, unit: str, pub: pd.DatetimeIndex | pd.Series, *, revision: int = 0,
                grade: PitGrade = PitGrade.A, estimated: bool = False) -> dict[str, object]:
    return {
        "publication_time": pd.DatetimeIndex(pub),
        "ingestion_time": pd.DatetimeIndex(pub) + pd.Timedelta(minutes=5),
        "revision": np.full(n, revision, dtype="int64"),
        "unit": [unit] * n,
        "source": [SYNTHETIC_SOURCE] * n,
        "quality_flag": [QualityFlag.SYNTHETIC.value] * n,
        "entitlement": [Entitlement.SYNTHETIC.value] * n,
        "pit_grade": [grade.value] * n,
        "publication_time_estimated": np.full(n, estimated),
        "source_timezone": [MARKET_TZ] * n,
    }


def storage_observations(cfg: SyntheticConfig) -> pd.DataFrame:
    """EU storage inventory (TWh) with a seasonal cycle, publication lag and some revisions.

    Returns OBSERVATION_SCHEMA rows for ``storage.eu.inventory_twh``: first releases plus revised
    vintages for a random subset of days (revision published 3 days after the first release).
    """
    rng = np.random.default_rng(cfg.seed + 1)
    days = pd.date_range(cfg.start, cfg.end, freq="D")
    doy = days.dayofyear.to_numpy()
    seasonal = 650 + 350 * np.sin(2 * np.pi * (doy - 120) / 365.25)
    shock = np.cumsum(rng.normal(0, 3.0, len(days)))
    shock -= pd.Series(shock).rolling(90, min_periods=1).mean().to_numpy()
    first = seasonal + shock
    event = _midnight_utc(days)
    pub = event + pd.Timedelta(days=cfg.storage_pub_lag_days, hours=7)
    base = pd.DataFrame(
        {
            "series_id": "storage.eu.inventory_twh",
            "event_time": event,
            "value": first,
            "geography": "EU",
            **_provenance(len(days), "TWh", pub, grade=PitGrade.B, estimated=True),
        }
    )
    revise = rng.random(len(days)) < cfg.storage_revision_prob
    rev = base.loc[revise].copy()
    rev["value"] = rev["value"] + rng.normal(0, 2.0, int(revise.sum()))
    rev["revision"] = 1
    rev["publication_time"] = rev["publication_time"] + pd.Timedelta(days=3)
    rev["ingestion_time"] = rev["publication_time"] + pd.Timedelta(minutes=5)
    return pd.concat([base, rev], ignore_index=True)


def _storage_surprise_by_day(storage: pd.DataFrame, cfg: SyntheticConfig) -> pd.Series:
    """First-release storage deviation from its trailing-365d same-season mean, by local date.
    Used only to plant signal; computed from first releases known at each publication."""
    storage = storage.loc[storage["series_id"] == cfg.planted_series_id]
    first = storage.loc[storage["revision"] == 0].set_index("event_time")["value"]
    if cfg.planted_driver == "change":
        first = first.diff()
    elif cfg.planted_driver != "level":
        raise ValueError(f"Unknown planted_driver {cfg.planted_driver!r}")
    dev = first - first.rolling(30, min_periods=10).mean()
    dev = dev / dev.rolling(250, min_periods=30).std()
    avail = storage.loc[storage["revision"] == 0].set_index("event_time")["publication_time"]
    by_pub_date = pd.Series(dev.to_numpy(), index=avail.dt.tz_convert(MARKET_TZ).dt.normalize())
    return by_pub_date.tz_localize(None).groupby(level=0).last()


def ttf_curve(cfg: SyntheticConfig, storage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Synthetic TTF monthly settlement curve in CURVE_SCHEMA (field ``settle``, EUR/MWh).

    One common log-price factor (random walk) plus a per-tenor seasonal premium; each contract is
    listed ``months_listed`` months ahead and stops trading ``expiry_bdays_before_delivery``
    business days before delivery. In the planted world the factor's daily return includes
    ``planted_beta`` × the storage surprise published by the previous day.
    """
    rng = np.random.default_rng(cfg.seed)
    days = trading_days(cfg)
    eps = rng.normal(0, cfg.daily_vol, len(days))
    if cfg.world == "planted":
        if storage is None:
            storage = storage_observations(cfg)
        surprise = _storage_surprise_by_day(storage, cfg).reindex(days, method="ffill")
        # Signal known at close of t-1 drives return on t: forecastable, not leaked.
        eps = eps - cfg.planted_beta * surprise.shift(1).fillna(0.0).to_numpy()
    elif cfg.world != "null":
        raise ValueError(f"Unknown synthetic world {cfg.world!r}")
    log_level = np.log(30.0) + np.cumsum(eps)

    rows: list[pd.DataFrame] = []
    first_month = (days[0] - pd.offsets.MonthBegin(1)).normalize()
    last_month = days[-1] + pd.DateOffset(months=cfg.months_listed)
    for start in pd.date_range(first_month, last_month, freq="MS"):
        last_trade = start - pd.offsets.BDay(cfg.expiry_bdays_before_delivery)
        listed_from = start - pd.DateOffset(months=cfg.months_listed)
        mask = (days >= listed_from) & (days <= last_trade)
        if not mask.any():
            continue
        d = days[mask]
        winter = 1.0 if start.month in (10, 11, 12, 1, 2, 3) else 0.0
        premium = 0.08 * winter
        idio = np.cumsum(rng.normal(0, cfg.contract_noise, int(mask.sum()))) if cfg.contract_noise else 0.0
        price = np.exp(log_level[mask] + premium + idio)
        n = len(d)
        end = start + pd.offsets.MonthBegin(1)
        rows.append(
            pd.DataFrame(
                {
                    "hub": "TTF",
                    "contract_id": contract_id("TTF", "M", start.strftime("%Y-%m")),
                    "tenor_type": "M",
                    "delivery_start": _midnight_utc(pd.DatetimeIndex([start] * n)),
                    "delivery_end": _midnight_utc(pd.DatetimeIndex([end] * n)),
                    "trade_date": _midnight_utc(d),
                    "field": "settle",
                    "value": price.astype(float),
                    "currency": "EUR",
                    **_provenance(n, "EUR/MWh", _settle_ts(d)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def weather_forecast_runs(cfg: SyntheticConfig, max_lead_days: int = 7) -> pd.DataFrame:
    """Daily 00z runs of population-weighted HDD for ``weather.eu.hdd`` (FORECAST_SCHEMA).

    Each valid day has one forecast per lead 1..max_lead_days; forecasts converge to the
    synthetic truth as lead shrinks. Run published 6 h after issue.
    """
    rng = np.random.default_rng(cfg.seed + 2)
    days = pd.date_range(cfg.start, cfg.end, freq="D")
    doy = days.dayofyear.to_numpy()
    truth = np.clip(9 + 8 * np.cos(2 * np.pi * (doy - 15) / 365.25) + rng.normal(0, 2, len(days)),
                    0, None)
    truth_s = pd.Series(truth, index=days)
    frames = []
    for lead in range(1, max_lead_days + 1):
        valid = days[lead:]
        issue = valid - pd.Timedelta(days=lead)
        noise = rng.normal(0, 0.4 * lead, len(valid))
        value = np.clip(truth_s.loc[valid].to_numpy() + noise, 0, None)
        n = len(valid)
        issue_utc = issue.tz_localize("UTC")
        frames.append(
            pd.DataFrame(
                {
                    "series_id": "weather.eu.hdd",
                    "issue_time": issue_utc,
                    "valid_time": valid.tz_localize("UTC"),
                    "lead_hours": float(lead * 24),
                    "model": "SYNTH_ENS",
                    "member": "mean",
                    "value": value,
                    "geography": "EU",
                    **_provenance(n, "degC_day", issue_utc + pd.Timedelta(hours=6)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
