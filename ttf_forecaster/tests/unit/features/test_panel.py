"""asof_panel must equal looping resolve_as_of + latest_value_per_series. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import DataError, PointInTimeViolation
from ttf_forecaster.data.synthetic import SyntheticConfig, storage_observations
from ttf_forecaster.features.panel import align_to_origins, asof_panel, final_vintage
from ttf_forecaster.features.point_in_time import latest_value_per_series, resolve_as_of


def _loop_reference(obs: pd.DataFrame, origins: pd.DatetimeIndex, sids: list[str],
                    staleness: pd.Timedelta | None, **kw: object) -> pd.DataFrame:
    rows = []
    for o in origins:
        resolved = resolve_as_of(obs, o, **kw)  # type: ignore[arg-type]
        s = latest_value_per_series(resolved, o, max_staleness=staleness) if len(resolved) else (
            pd.Series(dtype=float))
        rows.append(s.reindex(sids))
    return pd.DataFrame(rows, index=origins.rename("origin_time"), columns=sids).astype(float)


def _rich_observations() -> pd.DataFrame:
    """Synthetic storage (revisions + late publication) plus adversarial hand-made rows."""
    cfg = SyntheticConfig(start="2022-01-03", end="2022-06-30", storage_revision_prob=0.3)
    st = storage_observations(cfg)
    rng = np.random.default_rng(3)
    # second series: copy with different id, some values NaN, a grade-C subset, and a later
    # published *lower* revision (must not override).
    b = st.copy()
    b["series_id"] = "storage.de.inventory_twh"
    b.loc[rng.random(len(b)) < 0.05, "value"] = np.nan
    c_rows = b.sample(20, random_state=1).copy()
    c_rows["pit_grade"] = "C"
    c_rows["value"] = -999.0
    c_rows["publication_time"] = c_rows["publication_time"] - pd.Timedelta(days=1)
    late_low = b.loc[b["revision"] == 1].head(10).copy()
    late_low["revision"] = 0
    late_low["value"] = 12345.0
    late_low["publication_time"] = late_low["publication_time"] + pd.Timedelta(days=5)
    # a "schedule" row: event_time far after publication (visible early, usable only later)
    sched = b.iloc[[0]].copy()
    sched["event_time"] = pd.Timestamp("2022-07-15", tz="UTC")
    sched["publication_time"] = pd.Timestamp("2022-03-01", tz="UTC")
    sched["value"] = 7.0
    # exact (revision, publication_time) duplicate with the SAME value is legal (idempotent
    # re-ingestion); with different values it is a DataError (test_conflicting_vintages_raise).
    tie = b.loc[b["revision"] == 0].iloc[[40, 40]].copy()
    # (values left equal to the base row they duplicate)
    # sparse series (every 10th day): revisions land while the key is still the latest one,
    # so the revision logic is actually exercised by the latest-value panel.
    sp = st.loc[st["event_time"].dt.day.isin([1, 11, 21])].copy()
    sp["series_id"] = "storage.sparse_twh"
    sp_low = sp.loc[sp["revision"] == 1].copy()
    sp_low["revision"] = 0
    sp_low["value"] = 54321.0
    sp_low["publication_time"] = sp_low["publication_time"] + pd.Timedelta(days=1)
    sp_c = sp.loc[sp["revision"] == 0].head(8).copy()
    sp_c["pit_grade"] = "C"
    sp_c["revision"] = 5
    sp_c["value"] = -999.0
    out = pd.concat([st, b, c_rows, late_low, sched, tie, sp, sp_low, sp_c], ignore_index=True)
    out["ingestion_time"] = out["publication_time"] + pd.Timedelta(minutes=5)
    return out


def _origins() -> pd.DatetimeIndex:
    days = pd.bdate_range("2022-01-03", "2022-08-31")
    o = (days + pd.Timedelta(hours=19)).tz_localize("Europe/Amsterdam").tz_convert("UTC")
    # irregular extra origins straddling publication/ingestion instants
    extra = pd.DatetimeIndex(
        ["2022-01-07 07:02", "2022-02-10 07:00", "2022-03-01 00:00", "2022-01-02 00:00"], tz="UTC"
    )
    return o.append(extra)  # deliberately unsorted


@pytest.mark.parametrize("staleness", [None, pd.Timedelta(days=4)])
@pytest.mark.parametrize("strict", [False, True])
def test_equivalence_with_loop(staleness: pd.Timedelta | None, strict: bool) -> None:
    obs = _rich_observations()
    origins = _origins()
    sids = ["storage.eu.inventory_twh", "storage.de.inventory_twh", "storage.sparse_twh",
            "absent.series"]
    fast = asof_panel(obs, origins, sids, staleness, strict_ingestion=strict)
    ref = _loop_reference(obs, origins, sids, staleness, strict_ingestion=strict)
    pd.testing.assert_frame_equal(fast, ref, check_names=False)
    assert fast["absent.series"].isna().all()


def test_equivalence_all_grades() -> None:
    obs = _rich_observations()
    origins = _origins()
    sids = ["storage.de.inventory_twh", "storage.sparse_twh"]
    fast = asof_panel(obs, origins, sids, None, allowed_grades=None)
    ref = _loop_reference(obs, origins, sids, None, allowed_grades=None)
    pd.testing.assert_frame_equal(fast, ref, check_names=False)
    assert (fast.to_numpy() == -999.0).any()  # grade C visible only when admitted


def test_event_time_panel_and_grades() -> None:
    obs = _rich_observations()
    origins = _origins()
    vals, ev = asof_panel(obs, origins, ["storage.eu.inventory_twh"], return_event_time=True)
    assert vals.attrs["pit_grade"] == {"storage.eu.inventory_twh": "B"}
    ok = ev.notna().to_numpy().ravel()
    assert (ev.to_numpy().ravel()[ok] <= np.asarray(origins)[ok]).all()


def test_revisions_change_the_panel() -> None:
    """Guard against a vacuous equivalence test: revisions must matter on the sparse series."""
    obs = _rich_observations()
    sp = obs.loc[(obs["series_id"] == "storage.sparse_twh") & (obs["pit_grade"] != "C")]
    origins = _origins()
    full = asof_panel(sp, origins)
    first_only = asof_panel(sp.loc[sp["revision"] == 0], origins)
    differs = (full.fillna(0) != first_only.fillna(0)).to_numpy().sum()
    assert differs > 10
    assert not (full.to_numpy() == 54321.0).any()  # later-published lower revision never wins


def test_late_revision_not_seen_early() -> None:
    obs = _rich_observations()
    st = obs.loc[obs["series_id"] == "storage.eu.inventory_twh"]
    revised = st.loc[st["revision"] == 1].iloc[0]
    first = st.loc[(st["revision"] == 0) & (st["event_time"] == revised["event_time"])].iloc[0]
    before = revised["publication_time"] - pd.Timedelta(seconds=1)
    # at 'before' the key is still the latest one only if nothing newer was published; check via
    # the event panel and the resolver directly.
    v, ev = asof_panel(st, [before], return_event_time=True)
    res = resolve_as_of(st, before)
    row = res.loc[res["event_time"] == revised["event_time"]]
    assert row["value"].iloc[0] == first["value"]
    assert ev.iloc[0, 0] >= revised["event_time"]


def test_conflicting_vintages_raise_like_resolver() -> None:
    obs = _rich_observations()
    clash = obs.loc[(obs["series_id"] == "storage.eu.inventory_twh") & (obs["revision"] == 0)]
    clash = clash.iloc[[30]].copy()
    clash["value"] = clash["value"] + 1.0
    bad = pd.concat([obs, clash], ignore_index=True)
    o = _origins()
    with pytest.raises(DataError, match="conflicting values"):
        resolve_as_of(bad, o.max())
    with pytest.raises(DataError, match="conflicting values"):
        asof_panel(bad, o)
    # before the clashing rows are published neither path complains
    early = pd.DatetimeIndex([clash["publication_time"].iloc[0] - pd.Timedelta(hours=1)])
    asof_panel(bad, early)
    resolve_as_of(bad, early[0])


def test_naive_origins_rejected() -> None:
    with pytest.raises(PointInTimeViolation):
        asof_panel(_rich_observations(), pd.DatetimeIndex(["2022-02-01"]))


def test_align_to_origins_monotone_availability() -> None:
    et = pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-05"], tz="UTC")
    frame = pd.DataFrame({
        "event_time": et,
        # day 2 is available late (after day 3): day 3 must wait for it (cummax)
        "available_time": et + pd.to_timedelta([19, 60, 19], unit="h"),
        "x": [1.0, 2.0, 3.0],
    })
    origins = pd.DatetimeIndex(["2022-01-04 20:00", "2022-01-05 20:00", "2022-01-06 20:00"],
                               tz="UTC")
    out = align_to_origins(frame, origins, ["x"], max_staleness=pd.Timedelta(days=10))
    assert out["x"].tolist() == [1.0, 1.0, 3.0]


def test_final_vintage_uses_last_revision_publication() -> None:
    obs = _rich_observations()
    fv = final_vintage(obs.loc[obs["series_id"] == "storage.eu.inventory_twh"],
                       ["series_id", "event_time"])
    assert fv[["series_id", "event_time"]].duplicated().sum() == 0
    assert (fv["available_time"] == fv["publication_time"]).all()
