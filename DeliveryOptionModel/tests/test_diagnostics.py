"""Diagnostics on a small synthetic calibration: files written, numbers consistent."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DeliveryOptionModel.src.deliveryoption import calibration, diagnostics, snapshot
from DeliveryOptionModel.src.deliveryoption.market_data import hub_contract

AS_OF = pd.Timestamp("2025-06-30")
SPECS = {"TTF": {"future_root": "TZT", "option_root": "FJS", "fx": None, "mult": 1.0},
         "HH": {"future_root": "NG", "option_root": "NG", "fx": None, "mult": 1.0}}
HUBS = list(SPECS)
CAL = {"lookback_months": 12, "anchor_filter": "all", "return_days": 5,
       "vol_profile_buckets_days": [0, 30, 90, 365]}
PAIRS = [(pd.Period("2025-08", "M"), pd.Period("2025-09", "M")),
         (pd.Period("2025-09", "M"), pd.Period("2025-10", "M"))]


def synthetic_history():
    """Each hub: a common factor plus small per-contract noise -> high M/M+1 correlation."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", AS_OF)
    px, expiry = {}, {}
    for h, (vol, noise) in {"TTF": (0.5, 0.05), "HH": (0.4, 0.08)}.items():
        common = rng.normal(0, vol / np.sqrt(252), len(dates))
        for m in pd.period_range("2024-01", "2025-12", freq="M"):
            t = hub_contract(SPECS[h], m)
            fix = (m - 1).end_time.normalize() - pd.Timedelta(days=3)
            r = common + rng.normal(0, noise / np.sqrt(252), len(dates))
            s = pd.Series(50 * np.exp(np.cumsum(r)), index=dates)
            s[s.index > fix] = np.nan
            px[t], expiry[t] = s, fix
    return pd.DataFrame(px), pd.Series(expiry)


def build_calibration(folder):
    px, expiry = synthetic_history()
    fitted, summary = [], []
    for m0, m1 in PAIRS:
        fix = min(expiry[hub_contract(SPECS[h], m0)] for h in HUBS)
        window = (0, max((fix - AS_OF).days, 45))
        anchors = calibration.anchor_months(m0, AS_OF, CAL["lookback_months"], False)
        p = calibration.calibrate_pair(px, expiry, SPECS, HUBS, m0, m1, anchors, window, 5, AS_OF)
        fitted.append(p)
        summary.append({"pair": calibration.pair_key(m0, m1), "window_lo": window[0],
                        "window_hi": window[1], "n_obs": p.n_obs, "n_anchors": p.n_anchors,
                        "return_days": 5,
                        **{f"{h}_M/{h}_M1": p.corr.loc[f"{h}_M", f"{h}_M1"] for h in HUBS},
                        **{f"{h}_M/{h}_M1 daily": p.corr_daily.loc[f"{h}_M", f"{h}_M1"] for h in HUBS},
                        "TTF_M/HH_M": p.corr.loc["TTF_M", "HH_M"],
                        "TTF_M/HH_M daily": p.corr_daily.loc["TTF_M", "HH_M"],
                        "max_stale_share": float(p.stale_share.max())})
    rows = []
    for h in HUBS:
        for i, m in enumerate(pd.period_range("2025-08", "2025-10", freq="M")):
            fix = expiry[hub_contract(SPECS[h], m)]
            rows.append({"hub": h, "month": str(m), "future": hub_contract(SPECS[h], m), "fwd": 50.0,
                         "fixing": fix.date(), "lot_size": 1.0,
                         "atm_vol": [0.6, np.nan, np.nan][i] if h == "HH" else 0.5,
                         "vol": [0.6, 0.55, 0.5][i] if h == "HH" else 0.5,
                         "vol_source": "X [IVOL_MID]" if (h == "TTF" or i == 0) else "",
                         "option_expiry": (fix - pd.Timedelta(days=2)).date(), "note": ""})
    profile = calibration.vol_profile(px, expiry, SPECS, HUBS, CAL["vol_profile_buckets_days"], 5, AS_OF)
    snapshot.write(folder, AS_OF, pd.DataFrame(rows), {}, profile, fitted, pd.DataFrame(summary),
                   meta={"calibration": CAL, "hubs": SPECS})
    return px, expiry


def test_diagnostics_write_every_plot_and_table(tmp_path):
    folder = tmp_path / "2025-06-30"
    px, expiry = build_calibration(folder)
    with pytest.raises(FileNotFoundError, match="history"):
        diagnostics.run(folder, fetch=False)                      # nothing saved yet, no Bloomberg
    diagnostics.save_history(folder, px, expiry)
    out = diagnostics.run(folder, fetch=False)

    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == sorted(["vol_term_structure.png", "vol_profile.png", "corr_term_structure.png",
                           "corr_TTF_M_vs_TTF_M1.png", "corr_HH_M_vs_HH_M1.png",
                           "corr_TTF_M_vs_HH_M.png", "data_quality.png"])
    assert all((out / p).stat().st_size > 10_000 for p in pngs)
    for t in ("vol_term_structure", "vol_profile", "correlation_term_structure", "correlation_slices"):
        assert (out / "tables" / f"{t}.csv").exists()
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "HH vols:** 1 live mid, 0 settlement, 2 filled" in index


def test_slices_add_up_to_the_calibrated_correlation(tmp_path):
    folder = tmp_path / "2025-06-30"
    px, expiry = build_calibration(folder)
    snap = snapshot.read(folder)
    pairs = diagnostics.calibrated_pairs(folder)
    sl = diagnostics.correlation_slices(px, expiry, snap, pairs)
    summary = pd.read_csv(folder / "correlation_summary.csv").set_index("pair")
    for pair in summary.index:
        for series in ("TTF_M/TTF_M1", "HH_M/HH_M1", "TTF_M/HH_M"):
            pooled = sl[(sl.pair == pair) & (sl.series == series) & (sl.dim == "pooled")]
            assert pooled["corr"].iloc[0] == pytest.approx(summary.loc[pair, series], abs=1e-12)
            assert pooled["n"].iloc[0] == summary.loc[pair, "n_obs"]
            for dim in ("year", "days_to_expiry", "month_pair"):
                part = sl[(sl.pair == pair) & (sl.series == series) & (sl.dim == dim)]
                assert part["var_share"].sum() == pytest.approx(1.0)
                assert part["n"].sum() == summary.loc[pair, "n_obs"]


def test_vol_sources_are_classified():
    snap_like = type("S", (), {})()
    snap_like.as_of = AS_OF
    snap_like.vol_profile = pd.DataFrame({"hub": ["TTF"], "tau_lo_days": [0], "tau_hi_days": [np.inf],
                                          "realised_vol": [0.4], "n_obs": [10]})
    snap_like.contracts = pd.DataFrame({
        "hub": ["TTF"] * 4, "month": ["2025-08", "2025-09", "2025-10", "2025-11"], "future": list("abcd"),
        "fwd": 1.0, "atm_vol": [0.5, 0.4, np.nan, np.nan], "vol": [0.5, 0.4, 0.3, np.nan],
        "vol_source": ["x [IVOL_MID]", "y [IVOL_LAST]", "", ""],
        "option_expiry": pd.Timestamp("2025-09-01"), "fixing": pd.Timestamp("2025-09-03")}).set_index(["hub", "month"])
    vt = diagnostics.vol_table(snap_like)
    assert list(vt["source"]) == ["live mid", "settlement", "filled", "missing"]
    assert vt["realised_same_maturity"].iloc[0] == pytest.approx(0.4)
