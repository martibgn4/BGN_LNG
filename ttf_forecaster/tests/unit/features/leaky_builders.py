"""Deliberately broken builders for the leakage-canary and contract tests. SYNTHETIC only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import DataBundle
from ttf_forecaster.features import prices


def build_future_peek_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                               params: Mapping[str, Any]) -> pd.DataFrame:
    """LEAKS: uses the TTF settle of the *next* event date (a shift in the wrong direction)."""
    s = prices.front_return_series(bundle.curves, bundle.contracts, params)
    s["peek"] = s["front_settle"].shift(-1)
    frame = s.set_index("event_time")["peek"].sort_index()
    vals = frame.reindex(origins.floor("D"), method="ffill").to_numpy()
    out = pd.DataFrame({"future_peek__next_settle": vals}, index=origins)
    out.attrs["pit_grade"] = {"future_peek__next_settle": "A"}
    return out


def build_fullsample_z_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                                params: Mapping[str, Any]) -> pd.DataFrame:
    """LEAKS: z-scores a PIT-clean feature with full-sample mean/std (prohibition §5.3)."""
    f = prices.build_price_features(bundle, origins, params)["price__ret_1d"]
    z = (f - f.mean()) / f.std()
    out = pd.DataFrame({"fullz__ret_1d": z.to_numpy()}, index=origins)
    out.attrs["pit_grade"] = {"fullz__ret_1d": "A"}
    return out


def build_grade_c_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                           params: Mapping[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame({"gradec__x": np.ones(len(origins))}, index=origins)
    out.attrs["pit_grade"] = {"gradec__x": "C"}
    return out


def build_dupe_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                        params: Mapping[str, Any]) -> pd.DataFrame:
    """Breaks the contract: emits a column of another family."""
    out = pd.DataFrame({"calendar__weekday": np.zeros(len(origins))}, index=origins)
    out.attrs["pit_grade"] = {"calendar__weekday": "A"}
    return out


def provider_constant_returns(curves: pd.DataFrame, contracts: pd.DataFrame,
                              params: Mapping[str, Any]) -> pd.DataFrame:
    """A stand-in for contracts.rolls: the reference series with constant returns."""
    p = {k: v for k, v in params.items() if k != "return_series_provider"}
    s = prices.front_return_series(curves, contracts, p)
    s["log_return"] = 0.001
    return s


def build_notclean_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                            params: Mapping[str, Any]) -> pd.DataFrame:
    """Grade A but declares itself ineligible for a clean backtest (like the text family)."""
    out = pd.DataFrame({"notclean__x": np.ones(len(origins))}, index=origins)
    out.attrs["pit_grade"] = {"notclean__x": "A"}
    out.attrs["clean_backtest_eligible"] = False
    return out
