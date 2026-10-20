"""SYNTHETIC fixtures for the weather and power families. Nothing here is real data."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    CURVE_SCHEMA,
    DataBundle,
    empty_frame,
)
from ttf_forecaster.data.weather import (
    SYNTH_DET_MODEL,
    SYNTH_ENS_MODEL,
    SyntheticWeatherConfig,
    WeatherPoint,
    WeatherRegion,
    synthetic_point_weather,
)

POINTS = ("aa", "bb", "cc")


def bundle(observations: pd.DataFrame, forecasts: pd.DataFrame) -> DataBundle:
    return DataBundle(observations, forecasts, empty_frame(CURVE_SCHEMA),
                      empty_frame(CONTRACT_METADATA_SCHEMA))


def origins(start: str, end: str) -> pd.DatetimeIndex:
    """Business-day origins at 19:00 Europe/Amsterdam (after TTF settlement)."""
    return pd.bdate_range(start, end).tz_localize("Europe/Amsterdam") + pd.Timedelta(hours=19)


@pytest.fixture(scope="session")
def syn_points() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecasts, observations) for three synthetic points; obs from 2012 for normals."""
    cfg = SyntheticWeatherConfig(obs_start="2014-10-01", start="2022-12-01", end="2023-02-28",
                                 n_members=4)
    return synthetic_point_weather(POINTS, cfg)


@pytest.fixture()
def syn_params() -> dict[str, object]:
    """Builder params for the synthetic world: equal weights explicitly allowed (labelled)."""
    region = WeatherRegion("synthetic", "points",
                           tuple(WeatherPoint(p, p, None, None, None) for p in POINTS),
                           "requires_user_input")
    return {
        "region_obj": region,
        "allow_synthetic_equal_weights": True,
        "deterministic": {"model": SYNTH_DET_MODEL, "member": "det"},
        "ensemble": {"model": SYNTH_ENS_MODEL, "min_members": 3},
        "normals": {"min_years": 5},
    }


def _prov(n: int, pub: pd.DatetimeIndex, grade: str, unit: str) -> dict[str, object]:
    pub = pd.DatetimeIndex(pub)
    return {
        "publication_time": pub, "ingestion_time": pub, "revision": np.zeros(n, dtype="int64"),
        "unit": [unit] * n, "source": ["SYNTHETIC"] * n, "quality_flag": ["synthetic"] * n,
        "entitlement": ["synthetic"] * n, "pit_grade": [grade] * n,
        "publication_time_estimated": np.zeros(n, dtype=bool), "source_timezone": ["UTC"] * n,
    }


def make_obs(series_id: str, days: pd.DatetimeIndex, values: np.ndarray | float,
             lag: pd.Timedelta = pd.Timedelta(days=1), unit: str = "GW",
             grade: str = "B") -> pd.DataFrame:
    """Observation rows stamped at UTC midnight, published ``lag`` later."""
    ev = pd.DatetimeIndex(days).tz_localize("UTC") if pd.DatetimeIndex(days).tz is None else days
    n = len(ev)
    return pd.DataFrame({"series_id": series_id, "event_time": ev,
                         "value": np.broadcast_to(np.asarray(values, dtype=float), (n,)).copy(),
                         "geography": "XX", **_prov(n, ev + lag, grade, unit)})


def make_fc(series_id: str, issue_days: pd.DatetimeIndex, leads: range,
            value: Callable[[pd.Timestamp, int], float], unit: str = "GW",
            pub_lag: pd.Timedelta = pd.Timedelta(hours=6)) -> pd.DataFrame:
    """Daily forecast rows: one run per issue day (00z UTC), one row per lead (days)."""
    rows = []
    for issue in pd.DatetimeIndex(issue_days).tz_localize("UTC"):
        for ld in leads:
            rows.append((issue, issue + pd.Timedelta(days=ld), float(ld * 24), value(issue, ld)))
    df = pd.DataFrame(rows, columns=["issue_time", "valid_time", "lead_hours", "value"])
    n = len(df)
    df.insert(0, "series_id", series_id)
    df["model"] = "SYNTH"
    df["member"] = "det"
    df["geography"] = "XX"
    for k, v in _prov(n, pd.DatetimeIndex(df["issue_time"]) + pub_lag, "A", unit).items():
        df[k] = v
    return df
