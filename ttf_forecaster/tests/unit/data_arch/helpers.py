"""Small SYNTHETIC frames for data-architecture tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import SYNTHETIC_SOURCE


def obs_frame(
    rows: list[tuple[str, str, float, str, int]],
    *,
    unit: str = "TWh",
    grade: str = "A",
    tz: str = "UTC",
) -> pd.DataFrame:
    """OBSERVATION_SCHEMA rows from ``(series_id, event_time, value, publication_time, revision)``.

    Timestamps are parsed in ``tz`` and stored UTC. Labelled SYNTHETIC.
    """
    df = pd.DataFrame(rows, columns=["series_id", "event_time", "value", "publication_time",
                                     "revision"])
    n = len(df)
    for c in ("event_time", "publication_time"):
        df[c] = pd.to_datetime(df[c]).dt.tz_localize(tz).dt.tz_convert("UTC") if n else \
            pd.Series(dtype="datetime64[ns, UTC]")
        df[c] = df[c].astype("datetime64[ns, UTC]")
    df["ingestion_time"] = df["publication_time"]
    df["value"] = df["value"].astype(float)
    df["revision"] = df["revision"].astype("int64")
    for col, val in {"geography": "EU", "unit": unit, "source": SYNTHETIC_SOURCE,
                     "quality_flag": "synthetic", "entitlement": "synthetic",
                     "source_timezone": tz, "pit_grade": grade}.items():
        df[col] = pd.Series([val] * n, dtype=object)
    df["publication_time_estimated"] = np.zeros(n, dtype=bool)
    return df


def fc_frame(rows: list[tuple[str, str, str, float, str, int]]) -> pd.DataFrame:
    """FORECAST_SCHEMA rows from ``(series_id, issue, valid, value, publication, revision)`` (UTC)."""
    df = pd.DataFrame(rows, columns=["series_id", "issue_time", "valid_time", "value",
                                     "publication_time", "revision"])
    n = len(df)
    for c in ("issue_time", "valid_time", "publication_time"):
        df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
    df["lead_hours"] = ((df["valid_time"] - df["issue_time"]).dt.total_seconds() / 3600).astype(float)
    df["model"] = "SYNTH_ENS"
    df["member"] = "mean"
    df["ingestion_time"] = df["publication_time"]
    df["value"] = df["value"].astype(float)
    df["revision"] = df["revision"].astype("int64")
    for col, val in {"geography": "EU", "unit": "degC_day", "source": SYNTHETIC_SOURCE,
                     "quality_flag": "synthetic", "entitlement": "synthetic",
                     "source_timezone": "UTC", "pit_grade": "A"}.items():
        df[col] = [val] * n
    df["publication_time_estimated"] = np.zeros(n, dtype=bool)
    return df


def daily_obs(series_id: str, start: str, n: int, *, seed: int = 0, unit: str = "TWh",
              lag_days: int = 1) -> pd.DataFrame:
    """``n`` daily first-release observations (random walk), published ``lag_days`` later."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, periods=n, freq="D")
    vals = 100 + np.cumsum(rng.normal(0, 1, n))
    rows = [(series_id, str(d.date()), float(v), str((d + pd.Timedelta(days=lag_days)).date()), 0)
            for d, v in zip(days, vals, strict=True)]
    return obs_frame(rows, unit=unit)
