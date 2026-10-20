"""Shared target machinery: the curve context, label timing and TARGET_SCHEMA assembly.

Timing (``docs/target_definitions.md`` §0, §3):

* ``origin_time`` = publication time of the settlements of trading date ``t`` (default 19:00
  Europe/Amsterdam, ``configs/targets.yaml`` ``settlement_publication``), i.e. the close of ``t``.
* ``label_known_time`` = publication time of the last input the label needs: the settlement of the
  exit session for futures targets, the last DA publication / end of the last gas day for spot
  targets, the regime known-time for T9c.

Rows whose exit lies after the last available data are **not emitted** (the label does not exist
yet). Rows whose inputs are missing are emitted with ``value = NaN`` and
``quality_flag = missing_input``; nothing is interpolated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.calendars import TradingCalendar, to_local_day
from ttf_forecaster.contracts.futures import (
    ExpiryRule,
    contract_metadata,
    last_trade_map,
    parse_contract_id,
)
from ttf_forecaster.contracts.rolls import (
    MonthlyChain,
    RollRule,
    SettlementLookup,
    continuous_returns,
    settlement_panel,
)
from ttf_forecaster.data.base import (
    MARKET_TZ,
    SYNTHETIC_SOURCE,
    TARGET_SCHEMA,
    PublicationLag,
    QualityFlag,
    apply_publication_lag,
    empty_frame,
    validate_frame,
)
from ttf_forecaster.settings import get, load_config


def publication_time(day: pd.Timestamp, lag: PublicationLag, tz: str = MARKET_TZ) -> pd.Timestamp:
    """UTC publication time of an event on local date ``day`` under ``lag``."""
    ev = pd.Series([pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")])
    return pd.Timestamp(apply_publication_lag(ev, lag).iloc[0])


@dataclass
class CurveContext:
    """Everything a futures target needs: settlements, monthly chain, roll rule, timing.

    Build with :meth:`from_curve`. ``panel`` holds every tenor of ``hub`` (monthly contracts for
    the chain, listed Q/S contracts for strip fallbacks).
    """

    panel: pd.DataFrame
    chain: MonthlyChain
    rule: RollRule
    settle_lag: PublicationLag
    hub: str = "TTF"
    tz: str = MARKET_TZ
    is_synthetic: bool = False
    _pub_cache: dict[pd.Timestamp, pd.Timestamp] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.lookup = SettlementLookup(self.panel)

    @classmethod
    def from_curve(
        cls,
        curve: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        *,
        hub: str = "TTF",
        calendar: TradingCalendar | None = None,
        rule: RollRule | None = None,
        config_dir: str | None = None,
    ) -> CurveContext:
        """Context from a CURVE_SCHEMA frame.

        Args:
            curve: CURVE_SCHEMA rows (settle field) for ``hub``.
            metadata: CONTRACT_METADATA_SCHEMA rows; when None, last trading days are computed
                from the configured expiry rule on ``calendar``.
            calendar: defaults to the empirical calendar of ``curve`` (+ configured rule).
            rule: roll rule; defaults to ``configs/targets.yaml``.
        """
        cfg = load_config("targets", config_dir)
        field_name = str(cfg.get("field", "settle"))
        panel = settlement_panel(curve, hub=hub, field=field_name)
        cal = calendar or TradingCalendar.from_curve(curve, hub=hub, field=field_name,
                                                     config_dir=config_dir)
        if metadata is None:
            months = [parse_contract_id(c) for c in panel.columns]
            months = [c for c in months if c.tenor_type == "M"]
            metadata = contract_metadata(months, cal, ExpiryRule.from_config(hub, config_dir))
        chain = MonthlyChain(last_trade_map(metadata), cal)
        synthetic = bool(len(curve)) and bool((curve["source"] == SYNTHETIC_SOURCE).all())
        return cls(
            panel=panel,
            chain=chain,
            rule=rule or RollRule.from_config(config_dir),
            settle_lag=PublicationLag.from_config(get(cfg, "settlement_publication")),
            hub=hub,
            is_synthetic=synthetic,
        )

    @property
    def calendar(self) -> TradingCalendar:
        return self.chain.calendar

    def pub_time(self, day: pd.Timestamp) -> pd.Timestamp:
        """Settlement publication time (UTC) of local trading date ``day`` (cached)."""
        d = to_local_day(day)
        if d not in self._pub_cache:
            self._pub_cache[d] = publication_time(d, self.settle_lag, self.tz)
        return self._pub_cache[d]

    def default_origins(self) -> pd.DatetimeIndex:
        """Every session between the first and last panel date."""
        return self.calendar.sessions_in(self.lookup.first_day, self.lookup.last_day)

    def front_return_sigma(self, span: int, min_periods: int) -> pd.Series:
        """EWMA daily vol of the spliced front-month return series, known at each session close."""
        r = continuous_returns(self.panel, self.chain, self.rule.roll_days)["value"]
        var = (r**2).ewm(span=span, min_periods=min_periods, adjust=False, ignore_na=True).mean()
        return np.sqrt(var)


def normalise_origins(origins: Iterable[Any] | None, ctx: CurveContext) -> pd.DatetimeIndex:
    """Origins as naive local sessions. Non-session dates are rejected (no silent snapping)."""
    if origins is None:
        return ctx.default_origins()
    days = pd.DatetimeIndex([to_local_day(o) for o in origins])
    bad = [d.date() for d in days if not ctx.calendar.is_session(d)]
    if bad:
        raise ValueError(f"origins must be trading sessions; not sessions: {bad[:5]}")
    return days


class RowCollector:
    """Accumulates target rows and assembles a validated TARGET_SCHEMA frame."""

    def __init__(self, synthetic: bool = False) -> None:
        self.rows: list[dict[str, Any]] = []
        self.synthetic = synthetic

    def add(
        self,
        *,
        origin_time: pd.Timestamp,
        target_id: str,
        horizon: int,
        value: float,
        label_known_time: pd.Timestamp,
        contract_id: str | None,
        construction: str,
    ) -> None:
        v = float(value) if value is not None else float("nan")
        if not np.isfinite(v):
            flag = QualityFlag.MISSING_INPUT.value
            v = float("nan")
        else:
            flag = (QualityFlag.SYNTHETIC if self.synthetic else QualityFlag.OK).value
        self.rows.append(
            {
                "origin_time": origin_time,
                "target_id": target_id,
                "horizon": int(horizon),
                "value": v,
                "label_known_time": label_known_time,
                "contract_id": contract_id,
                "construction": construction,
                "quality_flag": flag,
            }
        )

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return empty_frame(TARGET_SCHEMA)
        df = pd.DataFrame(self.rows, columns=TARGET_SCHEMA.names)
        for col in ("origin_time", "label_known_time"):
            df[col] = pd.to_datetime(df[col], utc=True)
        df["horizon"] = df["horizon"].astype("int64")
        df["value"] = df["value"].astype(float)
        df["contract_id"] = df["contract_id"].astype(object)
        return validate_frame(df, TARGET_SCHEMA)


def concat_targets(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate TARGET_SCHEMA frames and re-validate."""
    frames = [f for f in frames if len(f)]
    if not frames:
        return empty_frame(TARGET_SCHEMA)
    out = pd.concat(frames, ignore_index=True)
    return validate_frame(out, TARGET_SCHEMA)


# ---------------------------------------------------------------------------------------------
# Session horizon (D-032 / F1)
# ---------------------------------------------------------------------------------------------


def label_overlap_rows(targets: pd.DataFrame) -> int:
    """Maximum number of LATER origins (rows) that precede an origin's ``label_known_time``.

    For T3 on a session grid this is ``h - 1``. For month-based targets (T4/T5) it is tens of
    sessions per month of ``horizon``. ``evaluation.diagnostics`` re-exports this function.
    """
    t = targets.dropna(subset=["value"]).sort_values("origin_time")
    o = pd.DatetimeIndex(t["origin_time"]).asi8
    k = pd.DatetimeIndex(t["label_known_time"]).asi8
    pos = np.searchsorted(o, k, side="left") - np.arange(len(o)) - 1
    return int(max(pos.max(), 0)) if len(pos) else 0


def session_horizon(targets: pd.DataFrame, declared: int | None = None) -> int:
    """Model horizon in trading sessions, derived from the label overlap: ``label_overlap_rows + 1``.

    Feeds the inner-CV gap, ``make_target_lags`` and the DM/CW HAC lag (``hac lag = h - 1``). The
    ``horizon`` column of a month-based target (T4/T5) is a month count and is never used here.

    Args:
        targets: one target x horizon TARGET_SCHEMA frame on a session grid.
        declared: the horizon the caller configured as sessions (T3). Refused if it disagrees with
            the label overlap, which is what a month count looks like.

    Raises:
        ValueError: when ``declared`` differs from the overlap-derived session horizon.
    """
    if targets.empty:
        raise ValueError("cannot derive a session horizon from an empty target frame")
    h = label_overlap_rows(targets) + 1
    if declared is not None and int(declared) != h:
        raise ValueError(
            f"declared horizon {declared} disagrees with the label overlap (session horizon {h}): "
            "month-based horizons (T4/T5) are refused here; use label_overlap_rows + 1 "
            "(D-032 F1)"
        )
    return h
