"""Forecast revisions: the signal itself.

    revision(issue_t, target_d) = forecast(issue_t, target_d)
                                - forecast(issue_{t-1}, target_d)

A difference along the ISSUE axis at a FIXED target date. The level of forecast
demand is public and priced; the change between two runs is the news (SPEC 0).

THE COMMON-COVERAGE RULE is the reason this module is not four lines long.

Consider a 7-day forecast on Monday covering Tue..Mon, and on Tuesday covering
Wed..Tue. Summing each and differencing gives a large number every single day,
because Tuesday's sum includes a target day Monday's never had and excludes one
it did. That number is not news - nothing was revised - but it is highly
seasonal, strongly autocorrelated, and would regress beautifully against
anything with a seasonal in it.

So a revision is computed ONLY over target days present in both issue dates.
Everything below enforces that, and `require_common_coverage: false` is
deliberately not implemented.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from wxreturns.config import get

log = logging.getLogger(__name__)

_KEYS = ["region", "issue_date", "target_date"]


def _lead_days(frame: pd.DataFrame) -> pd.Series:
    issue = pd.to_datetime(frame["issue_date"])
    target = pd.to_datetime(frame["target_date"])
    return (target - issue).dt.days


def historical_buckets() -> list[dict]:
    """Lead buckets that can actually be backfilled.

    Open-Meteo's previous-run archive stops at lead 7 (verified, see
    sources.yaml), so the far buckets exist only from the day live collection
    starts. Including them in a backtest would mean treating a missing far-lead
    revision as zero - an assertion that the 15-day forecast did not change.
    """
    return [b for b in get("features", "revisions", "lead_buckets")
            if b.get("historical", False)]


def all_buckets() -> list[dict]:
    return list(get("features", "revisions", "lead_buckets"))


def bucket_for_lead(lead: int, buckets: list[dict]) -> str | None:
    for bucket in buckets:
        if bucket["min_lead"] <= lead <= bucket["max_lead"]:
            return bucket["name"]
    return None


def daily_revisions(panel: pd.DataFrame, value_column: str = "hdd",
                    step_days: int = 1, historical_only: bool = True
                    ) -> pd.DataFrame:
    """Per-target-day revisions between issue dates ``step_days`` apart.

    This is the granular form: one row per (region, issue, target), carrying
    the change in the forecast for that target day. Aggregation to a contract
    window happens in `window_revisions`, which is where coverage bites.
    """
    frame = panel.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])

    if frame["issue_date"].isna().any():
        raise ValueError(
            "the panel contains reanalysis rows (issue_date is NaT). Those are "
            "actuals, not forecasts, and differencing them along an issue axis "
            "is meaningless. Filter to forecast rows first."
        )

    frame["lead"] = _lead_days(frame)
    buckets = historical_buckets() if historical_only else all_buckets()
    max_lead = max(b["max_lead"] for b in buckets)
    frame = frame[(frame["lead"] >= 0) & (frame["lead"] <= max_lead)]

    previous = frame.copy()
    previous["issue_date"] = previous["issue_date"] + pd.Timedelta(days=step_days)
    previous = previous.rename(columns={value_column: "previous_value"})

    merged = frame.merge(
        previous[_KEYS + ["previous_value"]], on=_KEYS, how="inner")

    if merged.empty:
        raise ValueError(
            f"no target day appears at two issue dates {step_days} day(s) "
            "apart, so no revision can be formed. The panel probably holds a "
            "single issue date per target - check that previous-run leads were "
            "actually requested."
        )

    merged["revision"] = merged[value_column] - merged["previous_value"]
    merged["step_days"] = step_days
    merged["bucket"] = merged["lead"].map(lambda x: bucket_for_lead(x, buckets))
    return merged.dropna(subset=["bucket"])


def window_revisions(panel: pd.DataFrame, windows: pd.DataFrame,
                     value_column: str = "hdd", step_days: int = 1,
                     historical_only: bool = True) -> pd.DataFrame:
    """Revisions aggregated over each contract's delivery window.

    ``windows`` has one row per (commodity, slot, obs_date) with the delivery
    window bounds - see `market/contracts.py`. The aggregate answers "how much
    did the forecast for THIS contract's gas change today", which is the
    quantity a front-month position is exposed to.

    COMMON COVERAGE. Only target days forecast at BOTH issue dates enter the
    sum, so a horizon rolling forward cannot masquerade as news. The count of
    common days and the coverage of the window both travel on the output.
    """
    if not get("features", "revisions", "require_common_coverage"):
        raise ValueError(
            "require_common_coverage is false. Differencing sums over unequal "
            "target sets produces a large, seasonal, entirely artificial "
            "'revision'. That path is not implemented."
        )

    granular = daily_revisions(panel, value_column, step_days, historical_only)
    min_coverage = get("features", "revisions", "min_coverage_fraction")

    windows = windows.copy()
    windows["window_start"] = pd.to_datetime(windows["window_start"])
    windows["window_end"] = pd.to_datetime(windows["window_end"])
    windows["obs_date"] = pd.to_datetime(windows["obs_date"])

    # Join every revision to the contracts live on its issue date, then keep
    # only revisions whose target falls inside that contract's window.
    joined = windows.merge(granular, left_on=["region", "obs_date"],
                           right_on=["region", "issue_date"], how="inner")
    if joined.empty:
        raise ValueError(
            "no overlap between the contract windows and the revision panel. "
            "Check that the weather region matches the commodity's region in "
            "targets.yaml, and that the date ranges intersect."
        )
    inside = joined[(joined["target_date"] >= joined["window_start"])
                    & (joined["target_date"] <= joined["window_end"])]
    if inside.empty:
        raise ValueError(
            "revisions exist but none fall inside a delivery window. With a "
            "7-day forecast horizon this is expected early in a month for the "
            "front+1 contract, whose window can be more than 7 days away."
        )

    keys = ["commodity", "slot", "obs_date", "region", "bucket"]
    aggregated = (inside.groupby(keys, as_index=False)
                  .agg(revision=("revision", "sum"),
                       days_common=("revision", "size"),
                       window_start=("window_start", "first"),
                       window_end=("window_end", "first")))

    span = ((aggregated["window_end"] - aggregated["window_start"]).dt.days
            + 1).astype(float)
    aggregated["coverage_fraction"] = aggregated["days_common"] / span
    aggregated["step_days"] = step_days
    aggregated["value_column"] = value_column

    usable = aggregated[aggregated["coverage_fraction"] >= min_coverage]
    dropped = len(aggregated) - len(usable)
    if dropped:
        log.info("dropped %d window-revision rows below the %s coverage floor",
                 dropped, min_coverage)
    if usable.empty:
        raise ValueError(
            f"every window revision fell below the {min_coverage} coverage "
            "floor. A 7-day forecast covers only a fraction of a calendar "
            "month, so either lower min_coverage_fraction deliberately or "
            "aggregate over balance-of-month instead of the full window."
        )
    return usable.reset_index(drop=True)


def pivot_buckets(window_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (commodity, slot, obs_date), one column per lead bucket.

    This is the design matrix shape. A bucket absent for a date becomes NaN and
    is NOT filled: a missing revision is missing information, and zero would
    assert the forecast did not change.
    """
    wide = window_frame.pivot_table(
        index=["commodity", "slot", "obs_date"],
        columns="bucket", values="revision", aggfunc="sum")
    wide.columns = [f"rev_{c}" for c in wide.columns]

    coverage = window_frame.pivot_table(
        index=["commodity", "slot", "obs_date"],
        columns="bucket", values="coverage_fraction", aggfunc="min")
    coverage.columns = [f"cov_{c}" for c in coverage.columns]

    return wide.join(coverage).reset_index()


def spread_features(spread: pd.DataFrame, step_days: int = 1) -> pd.DataFrame:
    """Ensemble spread and its change, per (region, issue, target).

    Spread is a VOLATILITY signal and is kept out of the mean equation for that
    reason (SPEC 0). It enters the variance model and scales position size.
    """
    frame = spread.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["lead"] = _lead_days(frame)

    if not get("features", "revisions", "ensemble", "include_spread_change"):
        return frame

    previous = frame.copy()
    previous["issue_date"] = previous["issue_date"] + pd.Timedelta(days=step_days)
    previous = previous.rename(columns={"spread": "previous_spread"})
    merged = frame.merge(previous[_KEYS + ["previous_spread"]],
                         on=_KEYS, how="left")
    merged["spread_change"] = merged["spread"] - merged["previous_spread"]
    return merged


def sanity_check_sign(revision_frame: pd.DataFrame,
                      cold_snap: tuple[str, str]) -> dict:
    """Gate 1 check: does a known cold snap show up as a positive HDD revision?

    Sign convention: HDD rises as temperature falls, so a forecast turning
    colder is a POSITIVE hdd revision. If a known cold event comes back
    negative the issue axis is inverted somewhere, and every coefficient
    downstream would carry the wrong sign.
    """
    start, end = pd.Timestamp(cold_snap[0]), pd.Timestamp(cold_snap[1])
    inside = revision_frame[(revision_frame["obs_date"] >= start)
                            & (revision_frame["obs_date"] <= end)]
    if inside.empty:
        raise ValueError(f"no revision rows between {start.date()} and {end.date()}")
    total = float(inside["revision"].sum())
    return {
        "window": f"{start.date()}..{end.date()}",
        "n_rows": int(len(inside)),
        "total_revision": total,
        "mean_revision": float(np.mean(inside["revision"])),
        "sign_as_expected": total > 0.0,
    }
