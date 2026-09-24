"""Build the design matrix from cached sources.

Two rules govern this module, and both exist to stop the model flattering
itself:

1. **Delivery month, not observation month.** The basis panel is two
   dimensional - observed on day t, for delivery month m. A seasonal fitted to
   t rather than m looks fine in-sample and forecasts nothing, because the
   thing that is seasonal is when the gas is delivered, not when it is quoted.
   Every join in here is on the delivery month.

2. **Publication lag.** Spark releases in the London afternoon. A feature
   stamped with today's release date is not knowable when today's forecast is
   made, so every feature is shifted by
   ``features.publication_lag_business_days`` before it enters the matrix.
   Skip this and the backtest reads tomorrow's news.

A feature whose source is unavailable is DROPPED and reported, never filled
with zero. ``build_panel`` returns the matrix and a manifest of what was
dropped and why, and the caller is expected to print the manifest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nwebasis import store
from nwebasis.config import get

log = logging.getLogger(__name__)

MONTHS_IN_YEAR = 12
_BASIS_DATASET = "spark_basis_nwe"
_SWE_DATASET = "spark_basis_swe"
_FREIGHT_ATLANTIC = "spark_freight_atlantic"
_FREIGHT_PACIFIC = "spark_freight_pacific"
_NETBACK_DATASET = "spark_netback_sabine"
_SLOT_DATASET = "spark_slots"


@dataclass
class PanelManifest:
    """What made it into the matrix, and what did not."""

    rows: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    included: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    def report(self) -> str:
        lines = [
            f"panel: {self.rows} rows, "
            f"{self.date_range[0].date()} to {self.date_range[1].date()}",
            f"features in:  {', '.join(self.included) or '(none)'}",
        ]
        if self.dropped:
            lines.append("features out:")
            lines += [f"  {name:<26} {reason}" for name, reason in self.dropped.items()]
        else:
            lines.append("features out: (none)")
        return "\n".join(lines)


def load_basis(dataset: str = _BASIS_DATASET) -> pd.DataFrame:
    """The target panel: one row per (release_date, delivery_start)."""
    frame = store.read_latest(dataset)
    keep = ["release_date", "tenor", "delivery_start", "value"]
    frame = frame[keep].dropna(subset=["value"])
    frame["release_date"] = pd.to_datetime(frame["release_date"])
    frame["delivery_start"] = pd.to_datetime(frame["delivery_start"])
    return frame.sort_values(["release_date", "delivery_start"]).reset_index(drop=True)


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    """Delivery-month number and its cyclical encoding.

    Both are provided on purpose. Twelve dummies fit the observed seasonal
    exactly but spend twelve degrees of freedom on ~2.5 years of data; the
    sine/cosine pair spends two and cannot represent the sharp February-March
    trough. Which is better is an empirical question the backtest answers, so
    the choice is a config switch rather than a decision buried here.
    """
    out = frame.copy()
    out["delivery_month"] = out["delivery_start"].dt.month
    angle = out["delivery_month"] * (np.pi * 2) / MONTHS_IN_YEAR
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    return out


def month_dummies(frame: pd.DataFrame) -> pd.DataFrame:
    """Eleven delivery-month dummies; January is the omitted base."""
    dummies = pd.get_dummies(frame["delivery_month"], prefix="dm", dtype=float)
    base = dummies.columns[0]
    return dummies.drop(columns=[base])


def _lag_business_days(frame: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Push a feature frame forward so it is knowable at forecast time."""
    lag = int(get("features", "publication_lag_business_days"))
    out = frame.copy()
    out[date_column] = out[date_column] + pd.tseries.offsets.BDay(lag)
    return out.sort_values(date_column)


def _asof_merge(panel: pd.DataFrame, feature: pd.DataFrame,
                by: str | None = None) -> pd.DataFrame:
    """Attach the most recent feature observation at or before each panel date.

    An exact-date merge looks right and quietly loses rows: the lagged feature
    date lands on a holiday, the basis was assessed that week anyway, and the
    row comes through as NaN. Two of those in a row and the differenced terms
    vanish too. As-of merging keeps the lag - the feature dates have already
    been pushed forward - while tolerating calendars that do not line up.
    """
    left = panel.sort_values("release_date")
    right = feature.sort_values("release_date")
    if by is not None:
        left = left.sort_values([ "release_date", by])
        right = right.sort_values(["release_date", by])
    merged = pd.merge_asof(left, right, on="release_date", by=by, direction="backward")
    return merged


def _daily_series(dataset: str, value_column: str, name: str) -> pd.Series:
    frame = store.read_latest(dataset)
    frame["release_date"] = pd.to_datetime(frame["release_date"])
    series = (frame.groupby("release_date")[value_column].mean()
              .sort_index().rename(name))
    return series


def attach_freight(panel: pd.DataFrame, manifest: PanelManifest) -> pd.DataFrame:
    """Atlantic and Pacific spot day rates, in CHANGES only.

    Levels are excluded deliberately. Over 2023-07-01..2026-09-03 the M+1 basis
    correlates +0.00 with Atlantic spot in levels and +0.38 in five-day changes.
    A level term would contribute no long-run information and a t-statistic
    manufactured by two trending series.
    """
    window = int(get("model", "ecm", "short_run_lags")) + 2
    try:
        atlantic = _daily_series(_FREIGHT_ATLANTIC, "usd_per_day", "freight_atlantic")
        pacific = _daily_series(_FREIGHT_PACIFIC, "usd_per_day", "freight_pacific")
    except FileNotFoundError as exc:
        manifest.dropped["atlantic_freight"] = f"not cached ({exc})"
        manifest.dropped["freight_basin_spread"] = "depends on freight"
        return panel

    freight = pd.concat([atlantic, pacific], axis=1).ffill()
    freight["d_freight_atlantic"] = freight["freight_atlantic"].diff(window)
    freight["d_freight_basin"] = (freight["freight_pacific"]
                                  - freight["freight_atlantic"]).diff(window)
    keep = freight[["d_freight_atlantic", "d_freight_basin"]].reset_index()
    keep = _lag_business_days(keep, "release_date")
    manifest.included += ["d_freight_atlantic", "d_freight_basin"]
    return _asof_merge(panel, keep)


def attach_netbacks(panel: pd.DataFrame, manifest: PanelManifest) -> pd.DataFrame:
    """East-west arb, plus the TTF and JKM levels that come with it.

    Joined on the DELIVERY month the netback implies, not on ``month_index``.
    Spark's netback is indexed by LOAD month and the cargo lands roughly six
    weeks later, so ``monthIdx`` and the basis curve's ``M+n`` do not line up.
    ``nweCargoDeliveryDate`` is what makes the two comparable, and using the
    index instead would misalign every row by about one month.
    """
    try:
        raw = store.read_latest(_NETBACK_DATASET)
    except FileNotFoundError as exc:
        for name in ("east_west_arb", "ttf_level", "ttf_time_spread"):
            manifest.dropped[name] = f"netbacks not cached ({exc})"
        return panel

    raw["release_date"] = pd.to_datetime(raw["release_date"])
    raw["delivery_start"] = (pd.to_datetime(raw["nwe_delivery_date"])
                             .dt.to_period("M").dt.to_timestamp())
    columns = ["release_date", "delivery_start", "arb_nea_minus_nwe",
               "route_cost_nwe", "route_cost_nea", "ttf_price", "jkm_price"]
    netbacks = (raw[columns].groupby(["release_date", "delivery_start"], as_index=False)
                .mean(numeric_only=True))

    # TTF term structure, from the netback payload's own TTF prints. This is
    # the fallback named in features.yaml; a Bloomberg TTF strip supersedes it.
    #
    # The spread is the CARRY OUT of each delivery month, TTF(m+1) - TTF(m),
    # not TTF(m) - TTF(m-1). Two reasons, one economic and one that cost an
    # hour of debugging:
    #
    #   Economically, "contango supports the prompt basis" is a statement about
    #   what the market pays to hold gas FROM month m INTO m+1.
    #
    #   Mechanically, a backward difference puts its NaN on the earliest month
    #   of each release. The netback payload starts about two months out, so
    #   that NaN lands on exactly the rows an M+2 forecast needs, and the
    #   feature silently survives with 78 of 804 rows populated - enough to fit
    #   on, not enough to mean anything. A forward difference moves the gap to
    #   the far end of the curve, where nothing is being forecast.
    #
    # Columns are reindexed to a dense monthly range first so that a shift of
    # one column is always a shift of one month, even if some release happens
    # to skip a delivery month.
    ttf = netbacks.pivot(index="release_date", columns="delivery_start", values="ttf_price")
    months = pd.date_range(ttf.columns.min(), ttf.columns.max(), freq="MS")
    ttf = ttf.reindex(columns=months)
    carry = ttf.shift(-1, axis=1) - ttf
    ttf_spread = carry.stack(future_stack=True).rename("ttf_time_spread").reset_index()
    ttf_spread.columns = ["release_date", "delivery_start", "ttf_time_spread"]

    netbacks = netbacks.merge(ttf_spread, on=["release_date", "delivery_start"], how="left")
    netbacks = _lag_business_days(netbacks, "release_date")
    manifest.included += ["arb_nea_minus_nwe", "route_cost_nwe", "ttf_time_spread"]
    return _asof_merge(panel, netbacks, by="delivery_start")


def attach_slots(panel: pd.DataFrame, manifest: PanelManifest) -> pd.DataFrame:
    """Free NWE regas slots for the matching delivery month.

    The slot file is wide - one column per M+n bucket with a parallel start
    date column - so it is melted into (release_date, delivery_start, slots)
    before joining. Counts are summed across NWE terminals.

    SIGN: this counts AVAILABLE slots. Falling means tightening, so the
    expected coefficient on the basis is POSITIVE - more free slots, easier to
    place a cargo, narrower discount.
    """
    try:
        raw = store.read_latest(_SLOT_DATASET)
    except FileNotFoundError as exc:
        manifest.dropped["regas_slot_scarcity"] = f"not cached ({exc})"
        return panel

    raw["ReleaseDate"] = pd.to_datetime(raw["ReleaseDate"])
    buckets = [c for c in raw.columns if c.startswith("M+") and not c.endswith("StartDate")]
    records = []
    for bucket in buckets:
        start_column = f"{bucket}StartDate"
        if start_column not in raw.columns:
            continue
        piece = raw[["ReleaseDate", "TerminalCode", bucket, start_column]].copy()
        piece.columns = ["release_date", "terminal", "slots", "delivery_start"]
        records.append(piece)
    if not records:
        manifest.dropped["regas_slot_scarcity"] = "slot file carried no M+n buckets"
        return panel

    melted = pd.concat(records, ignore_index=True)
    melted["delivery_start"] = pd.to_datetime(melted["delivery_start"])
    melted = melted.dropna(subset=["slots"])
    slots = (melted.groupby(["release_date", "delivery_start"], as_index=False)["slots"]
             .sum().rename(columns={"slots": "free_slots"}))
    slots["log_free_slots"] = np.log1p(slots["free_slots"])
    slots = _lag_business_days(slots, "release_date")
    manifest.included += ["log_free_slots"]
    return _asof_merge(panel, slots, by="delivery_start")


def attach_swe(panel: pd.DataFrame, manifest: PanelManifest) -> pd.DataFrame:
    """SWE minus NWE basis, same delivery month - the congestion tell."""
    try:
        swe = store.read_latest(_SWE_DATASET)
    except FileNotFoundError as exc:
        manifest.dropped["swe_nwe_spread"] = f"not cached ({exc})"
        return panel

    swe["release_date"] = pd.to_datetime(swe["release_date"])
    swe["delivery_start"] = pd.to_datetime(swe["delivery_start"])
    swe = (swe[["release_date", "delivery_start", "value"]]
           .rename(columns={"value": "swe_basis"}))
    swe = _lag_business_days(swe, "release_date")
    merged = _asof_merge(panel, swe, by="delivery_start")
    merged["swe_nwe_spread"] = merged["swe_basis"] - merged["value"]
    manifest.included += ["swe_nwe_spread"]
    return merged


def build_panel(tenors: list[str] | None = None) -> tuple[pd.DataFrame, PanelManifest]:
    """Assemble the full design matrix from whatever is cached."""
    tenors = tenors or list(get("target", "tenors", "fitted"))
    basis = load_basis()
    basis = basis[basis["tenor"].isin(tenors)]
    panel = add_calendar(basis)

    manifest = PanelManifest(
        rows=len(panel),
        date_range=(panel["release_date"].min(), panel["release_date"].max()),
        included=["month_sin", "month_cos"],
    )
    for name in ("storage_headroom", "lng_on_water", "weather_forecast_revision"):
        blocker = get("features", "long_run", name, "blocked_by") if name != \
            "weather_forecast_revision" else get("features", "short_run", name, "blocked_by")
        manifest.dropped[name] = blocker

    panel = attach_freight(panel, manifest)
    panel = attach_netbacks(panel, manifest)
    panel = attach_slots(panel, manifest)
    panel = attach_swe(panel, manifest)

    panel = panel.sort_values(["tenor", "release_date"]).reset_index(drop=True)
    manifest.rows = len(panel)
    manifest.date_range = (panel["release_date"].min(), panel["release_date"].max())
    return panel, manifest
