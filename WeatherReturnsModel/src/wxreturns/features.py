"""Assemble the design matrix, with the publication lag applied.

THE PUBLICATION LAG is the discipline this module exists to enforce. A 00z run
is not downloadable at 00:00 - ECMWF's 0.25 degree output lands roughly eight
hours later. A feature built from a run and stamped at that run's nominal time
is a feature from the future, and a backtest using it looks superb.

So every forecast row is stamped with the first market close at or after
``issue_time + available_after_hours``, and features join to returns on THAT
date. `sources.yaml` carries the lag per model.

The second rule here: a missing feature is missing, never zero. A zero revision
asserts the forecast did not change, which is a strong claim and usually a
false one. Rows with missing features are dropped, loudly.
"""

from __future__ import annotations

import logging

import pandas as pd

from wxreturns.config import get
from wxreturns.weather import revisions

log = logging.getLogger(__name__)

_HOURS_PER_DAY = 24


def tradeable_date(issue_dates: pd.Series, model: str,
                   calendar: pd.DatetimeIndex) -> pd.Series:
    """The first session at which a run could actually have been traded.

    ``calendar`` is the set of dates on which the market settled, so a run
    landing on a Saturday maps to Monday rather than to a date with no price.
    """
    lag_hours = get("sources", "open_meteo", "models", model,
                    "available_after_hours")
    available = pd.to_datetime(issue_dates) + pd.Timedelta(hours=lag_hours)
    # Normalising forward: a run available at 08:00 is tradeable that session;
    # one available after the close would need the next, which is why the
    # ceiling is taken against the settlement calendar rather than by rounding.
    ceiling = available.dt.ceil("D") if lag_hours % _HOURS_PER_DAY else available

    ordered = pd.DatetimeIndex(sorted(calendar))
    positions = ordered.searchsorted(ceiling.to_numpy(), side="left")
    positions = positions.clip(max=len(ordered) - 1)
    return pd.Series(ordered[positions], index=issue_dates.index)


def revision_features(panel: pd.DataFrame, windows: pd.DataFrame,
                      value_columns: tuple[str, ...] = ("hdd", "cdd"),
                      historical_only: bool = True) -> pd.DataFrame:
    """Window revisions for each value column, widened into feature columns."""
    steps = get("features", "revisions", "issue_steps_days")

    pieces: list[pd.DataFrame] = []
    for column in value_columns:
        for step in steps:
            try:
                windowed = revisions.window_revisions(
                    panel, windows, value_column=column, step_days=step,
                    historical_only=historical_only)
            except ValueError as exc:
                log.warning("no %s revisions at step %d: %s", column, step, exc)
                continue
            wide = revisions.pivot_buckets(windowed)
            rename = {c: f"{column}_{c}_s{step}"
                      for c in wide.columns if c.startswith("rev_")}
            pieces.append(wide.rename(columns=rename)
                          .drop(columns=[c for c in wide.columns
                                         if c.startswith("cov_")]))

    if not pieces:
        raise ValueError(
            "no revision features could be built. The usual cause is that the "
            "panel holds one issue date per target, so nothing can be "
            "differenced - check that previous-run leads were requested.")

    merged = pieces[0]
    for piece in pieces[1:]:
        merged = merged.merge(piece, on=["commodity", "slot", "obs_date"],
                              how="outer")
    return merged


def assemble(revision_frame: pd.DataFrame, returns: pd.DataFrame,
             level_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join features to forward returns on (commodity, slot, obs_date).

    Anomaly LEVELS join here as controls, never as the signal (SPEC 0). They
    are present so the revision coefficient is not picking up a seasonal level
    effect that has nothing to do with news.
    """
    keys = ["commodity", "slot", "obs_date"]
    frame = revision_frame.copy()
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])

    if level_frame is not None:
        levels = level_frame.copy()
        levels["obs_date"] = pd.to_datetime(levels["obs_date"])
        frame = frame.merge(levels, on=keys, how="left")

    target = returns.copy()
    target["obs_date"] = pd.to_datetime(target["obs_date"])
    merged = frame.merge(target, on=keys, how="inner")

    if merged.empty:
        raise ValueError(
            "features and returns share no (commodity, slot, obs_date) rows. "
            "The usual cause is the publication lag not having been applied, "
            "so features are stamped on dates the market did not settle.")
    return merged


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The model inputs: revisions, and anomaly levels as controls."""
    return [c for c in frame.columns
            if c.startswith(("hdd_rev_", "cdd_rev_")) or c.endswith("_anomaly")]


def drop_incomplete(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop rows with any missing feature, and say how many went.

    Not filled with zero. A zero revision is a claim that the forecast did not
    change, and filling with the column mean is a claim that it changed by a
    typical amount - both are inventions, and both would be invisible in the
    output.
    """
    if get("features", "hygiene", "raise_on_missing"):
        before = len(frame)
        clean = frame.dropna(subset=columns)
        dropped = before - len(clean)
        if dropped:
            log.warning("dropped %d of %d rows with incomplete features "
                        "(%.1f%%)", dropped, before,
                        dropped / before * float(_PERCENT))
        if clean.empty:
            raise ValueError(
                f"every row is missing at least one of {columns}. Check that "
                "all lead buckets requested are actually available - the "
                "historical archive stops at lead 7.")
        return clean
    return frame


_PERCENT = 100
