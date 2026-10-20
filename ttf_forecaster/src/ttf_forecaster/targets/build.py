"""Build every configured target T1-T9 (``configs/targets.yaml``) into one TARGET_SCHEMA frame.

T1/T4/T5b need a day-ahead frame and T9c needs a regime series; when those inputs are absent the
targets are skipped and listed in ``frame.attrs["skipped"]`` - never fabricated.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ttf_forecaster.data.base import PublicationLag
from ttf_forecaster.settings import get, load_config
from ttf_forecaster.targets.common import CurveContext, concat_targets
from ttf_forecaster.targets.futures_targets import (
    build_delivery_excess_futures,
    build_direction,
    build_m1_logret,
    build_m1_settle,
    build_realised_vol,
    build_regime_change,
    build_spread_change,
    build_strip_logret,
    build_tail,
)
from ttf_forecaster.targets.spot_targets import (
    build_da_level,
    build_da_month_avg,
    build_delivery_excess_physical,
)

log = logging.getLogger(__name__)


def build_targets(
    ctx: CurveContext,
    *,
    origins: Iterable[pd.Timestamp] | None = None,
    da_observations: pd.DataFrame | None = None,
    regimes: pd.Series | None = None,
    regime_known_times: pd.Series | None = None,
    config_dir: str | None = None,
) -> pd.DataFrame:
    """All targets in ``configs/targets.yaml`` for ``origins`` (default: every session)."""
    cfg = load_config("targets", config_dir)
    t: dict[str, Any] = get(cfg, "targets")
    ann = int(cfg.get("annualisation_days", 252))
    days = None if origins is None else list(origins)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    frames.append(build_m1_settle(ctx, t["T2"]["horizons"], origins=days, target_id=t["T2"]["id"]))
    frames.append(build_m1_logret(ctx, t["T3"]["horizons"], origins=days, target_id=t["T3"]["id"]))
    t5 = t["T5"]
    frames.append(build_delivery_excess_futures(
        ctx, t5["ks"], origins=days, target_id=t5["ids"]["futures"],
        log_target_id=t5["ids"]["futures_log"]))
    frames.append(build_strip_logret(ctx, t["T6"]["quarter"]["horizons"], "Q", origins=days,
                                     target_id=t["T6"]["quarter"]["id"]))
    frames.append(build_strip_logret(ctx, t["T6"]["season"]["horizons"], "S", origins=days,
                                     target_id=t["T6"]["season"]["id"]))
    for which, spec in t["T7"].items():
        frames.append(build_spread_change(ctx, spec["horizons"], which, origins=days,
                                          target_id=spec["id"]))
    frames.append(build_realised_vol(ctx, t["T8"]["horizons"], annualisation=ann, origins=days,
                                     target_id=t["T8"]["id"]))
    t9 = t["T9"]
    frames.append(build_direction(ctx, t9["direction"]["horizons"], origins=days,
                                  target_id=t9["direction"]["id"]))
    tail = t9["tail"]
    frames.append(build_tail(ctx, tail["horizons"], k=float(tail["k"]), mode=str(tail["mode"]),
                             ewma_span=int(tail["ewma_span"]),
                             ewma_min_periods=int(tail["ewma_min_periods"]),
                             fixed_threshold=float(tail["fixed_threshold"]), origins=days,
                             target_id=tail["id"]))

    if da_observations is None:
        skipped += ["T1", "T4", "T5b"]
    else:
        t1, t4 = t["T1"], t["T4"]
        frames.append(build_da_level(
            ctx, da_observations, series_id=t1["da_series_id"],
            assessment_session_offset=int(t1["assessment_session_offset"]),
            publication_fallback=PublicationLag.from_config(t1["publication_fallback"]),
            origins=days, target_id=t1["id"]))
        every = bool(t4["require_every_calendar_gas_day"])
        frames.append(build_da_month_avg(ctx, da_observations, t4["ks"],
                                         series_id=t4["da_series_id"],
                                         require_every_calendar_gas_day=every, origins=days,
                                         target_id=t4["id"]))
        frames.append(build_delivery_excess_physical(
            ctx, da_observations, t5["ks"], series_id=t4["da_series_id"],
            require_every_calendar_gas_day=every, origins=days,
            target_id=t5["ids"]["physical"], log_target_id=t5["ids"]["physical_log"]))

    if regimes is None:
        skipped.append("T9c")
    else:
        rc = t9["regime_change"]
        frames.append(build_regime_change(ctx, regimes, rc["horizons"],
                                          known_times=regime_known_times, origins=days,
                                          target_id=rc["id"]))
    if skipped:
        log.info("targets skipped for missing inputs: %s", skipped)
    out = concat_targets(frames)
    out.attrs["skipped"] = skipped
    return out
