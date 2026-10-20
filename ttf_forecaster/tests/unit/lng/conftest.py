"""SYNTHETIC fixtures for the LNG family. Nothing here is evidence about TTF or LNG."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import (
    OBSERVATION_SCHEMA,
    SYNTHETIC_SOURCE,
    DataBundle,
    PublicationLag,
    apply_publication_lag,
    validate_frame,
)
from ttf_forecaster.data.spark import MockSparkAdapter, load_spark_config
from ttf_forecaster.settings import load_config

START, END = date(2022, 3, 1), date(2022, 6, 30)


def bloomberg_lng_observations(start: date = START, end: date = END, seed: int = 3
                               ) -> pd.DataFrame:
    """SYNTHETIC rows for every series in bloomberg_lng.yaml, using its lag rules and grades.
    Transit counts are small Poisson integers; AHOY legs are zero on most days (as in reality)."""
    cfg = load_config("sources/bloomberg_lng")
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    event = days.tz_localize("Europe/Amsterdam").tz_convert("UTC")
    frames = []
    for sid, e in cfg["series"].items():
        n = len(days)
        if "transits" in sid:
            v = rng.poisson(0.7, n).astype(float)
        elif ".flow_" in sid:
            v = np.where(rng.random(n) < 0.6, 0.0, rng.uniform(5e4, 8e4, n))
        elif "vessels" in sid:
            v = np.round(np.clip(10 + np.cumsum(rng.normal(0, 1, n)), 0, None))
        else:
            v = 1e6 + np.cumsum(rng.normal(0, 2e4, n))
        pub = apply_publication_lag(pd.Series(event), PublicationLag.from_config(
            e["publication_lag"]))
        frames.append(pd.DataFrame({
            "series_id": sid, "event_time": event, "value": v.astype(float),
            "geography": str(e["geography"]), "publication_time": pub.to_numpy(),
            "ingestion_time": pub.to_numpy(), "revision": np.zeros(n, dtype="int64"),
            "unit": e["unit"], "source": SYNTHETIC_SOURCE, "quality_flag": "synthetic",
            "entitlement": "synthetic", "pit_grade": e["pit_grade"],
            "publication_time_estimated": np.ones(n, dtype=bool),
            "source_timezone": "Europe/Amsterdam"}))
    out = pd.concat(frames, ignore_index=True)
    for c in ("publication_time", "ingestion_time"):
        out[c] = pd.to_datetime(out[c], utc=True)
    return validate_frame(out, OBSERVATION_SCHEMA)


@pytest.fixture(scope="module")
def spark_cfg() -> dict:
    load_config.cache_clear()
    return load_spark_config()


@pytest.fixture(scope="module")
def mock_adapter(spark_cfg: dict) -> MockSparkAdapter:
    return MockSparkAdapter(spark_cfg, world_start="2022-01-01", world_end="2022-07-31")


@pytest.fixture(scope="module")
def lng_bundle(mock_adapter: MockSparkAdapter) -> DataBundle:
    """Bundle with SYNTHETIC Bloomberg LNG aggregates, Spark netbacks/slots, vessel-derived
    statistics, realised arrivals and PIT arrival forecasts."""
    route_ids = ["route.sabine_pass.arb_nea_minus_nwe_mi1_usd_mmbtu",
                 "lng.slots.syn_grain.available_m1_count", "lng.slots.syn_gate.available_m1_count"]
    obs = pd.concat([
        bloomberg_lng_observations(),
        mock_adapter.get_route_economics(route_ids, START, END),
        mock_adapter.get_destination_statistics(START, END),
        mock_adapter.get_realised_arrivals(date(2022, 1, 1), END),
    ], ignore_index=True)
    fc = mock_adapter.get_lng_arrivals(START, END)
    b = DataBundle.empty()
    return DataBundle(obs, fc, b.curves, b.contracts)


@pytest.fixture(scope="module")
def origins() -> pd.DatetimeIndex:
    local = pd.bdate_range("2022-04-01", "2022-06-30") + pd.Timedelta(hours=19)
    return local.tz_localize("Europe/Amsterdam").tz_convert("UTC")
