"""Calibration -> files -> pricing, end to end on synthetic data. No Bloomberg."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from DeliveryOptionModel.src.deliveryoption import calibration, deal, price_job, snapshot
from DeliveryOptionModel.src.deliveryoption.calibration import PairCalibration, leg_names
from DeliveryOptionModel.src.deliveryoption.market_data import contract_ticker
from DeliveryOptionModel.src.deliveryoption.vols import horizon_vol

AS_OF = pd.Timestamp("2026-09-23")
HUB_SPECS = {
    "TTF": {"future_root": "TZT", "option_root": "FJS", "fx": "EUR Curncy", "mult": 1 / 3.412141633},
    "HH": {"future_root": "NG", "option_root": "NG", "fx": None, "mult": 1.0},
}
HUBS = list(HUB_SPECS)
PRICING = {"vol_horizon": "samuelson", "n_paths": 40_000, "seed": 3, "bump_vol": 0.01,
           "bump_corr": 0.01, "output_dir": "output",
           "grid": {"corr_legs": ["TTF_M", "TTF_M1"], "corr": [0.9, 0.99], "vol_scale": [1.0]}}
MARKET = {"hubs": HUB_SPECS, "rate": 0.04, "pricing": PRICING}
PROFILE = pd.DataFrame({"hub": ["TTF", "TTF", "HH", "HH"], "tau_lo_days": [0, 30, 0, 30],
                        "tau_hi_days": [30, np.inf, 30, np.inf],
                        "realised_vol": [0.9, 0.6, 0.6, 0.45], "n_obs": [100] * 4})


# ---------------------------------------------------------------------------
# horizon vol
# ---------------------------------------------------------------------------

def test_flat_profile_leaves_the_vol_alone():
    flat = PROFILE.assign(realised_vol=0.5)
    for h in (5, 20, 40, 58):
        assert horizon_vol(0.8, 58, 60, h, flat[flat.hub == "TTF"]) == pytest.approx(0.8)


def test_samuelson_strips_the_front_month_out():
    """Front bucket vol 1.0, beyond 0.5: fixing in 60d, option in 58d, horizon 30d."""
    p = pd.DataFrame({"tau_lo_days": [0, 30], "tau_hi_days": [30, np.inf], "realised_vol": [1.0, 0.5]})
    # option covers tau in [2, 60]: 28*1 + 30*0.25 = 35.5; horizon covers [30, 60]: 7.5
    expected = 0.8 * np.sqrt(58 / 30 * 7.5 / 35.5)
    assert horizon_vol(0.8, 58, 60, 30, p) == pytest.approx(expected)
    assert horizon_vol(0.8, 58, 60, 58, p) == pytest.approx(0.8)


def test_vol_profile_recovers_bucket_vols():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-02", "2026-06-30")
    px, expiry = {}, {}
    for m in pd.period_range("2024-01", "2026-06", freq="M"):
        t = contract_ticker("TZT", m)
        fix = (m - 1).end_time.normalize()
        tau = (fix - dates).days.to_numpy()
        vol = np.where(tau < 30, 0.8, 0.4)
        px[t] = np.exp(np.cumsum(rng.normal(0, vol / np.sqrt(252))))
        px[t][tau < 0] = np.nan
        expiry[t] = fix
    prof = calibration.vol_profile(pd.DataFrame(px, index=dates), pd.Series(expiry),
                                   {"TTF": {"future_root": "TZT"}}, ["TTF"], [0, 30, 90], 5, AS_OF)
    got = prof.set_index("tau_lo_days")["realised_vol"]
    assert got[0] == pytest.approx(0.8, rel=0.1)
    assert got[30] == pytest.approx(0.4, rel=0.1)


# ---------------------------------------------------------------------------
# snapshot + pricing job
# ---------------------------------------------------------------------------

def synthetic_calibration(folder):
    months = [pd.Period(m, "M") for m in ("2026-11", "2026-12", "2027-01")]
    fwd = {"TTF": [73.0, 72.9, 72.0], "HH": [3.18, 3.51, 3.90]}
    vol = {"TTF": [0.89, 0.96, 0.90], "HH": [0.51, 0.54, 0.55]}
    rows = []
    for h in HUBS:
        for i, m in enumerate(months):
            fix = (m - 1).end_time.normalize() - pd.Timedelta(days=2)
            rows.append({"hub": h, "month": str(m), "future": contract_ticker(HUB_SPECS[h]["future_root"], m),
                         "fwd": fwd[h][i], "fixing": fix.date(), "lot_size": 720.0 if h == "TTF" else 10000.0,
                         "atm_vol": vol[h][i], "vol_source": "synthetic",
                         "option_expiry": (fix - pd.Timedelta(days=2)).date(), "note": "", "vol": vol[h][i]})
    names = leg_names(HUBS)
    c = pd.DataFrame(np.array([[1, .985, .3, .3], [.985, 1, .3, .3],
                               [.3, .3, 1, .94], [.3, .3, .94, 1.0]]), names, names)
    pairs = [PairCalibration(m0=a, m1=b, corr=c, corr_daily=c, realised_vol=pd.Series(0.6, names),
                             n_obs=100, n_anchors=20, anchors=[], window_days=(0, 45), return_days=5,
                             stale_share=pd.Series(0.0, names))
             for a, b in zip(months, months[1:])]
    summary = pd.DataFrame({"pair": [calibration.pair_key(p.m0, p.m1) for p in pairs]})
    snapshot.write(folder, AS_OF, pd.DataFrame(rows), {"EUR Curncy": 1.1414}, PROFILE, pairs,
                   summary, meta={})
    return snapshot.read(folder)


TRADES = {
    "defaults": {"volume_mmbtu": 3_600_000, "sell": {"hub": "HH", "slope": 1.0},
                 "buy": {"hub": "TTF", "slope": 1.0}, "strike_usd_mmbtu": 21.5,
                 "can_cancel": False, "decision_date": None, "pay_lag_days": 20},
    "trades": [
        {"id": "NovDec", "months": ["2026-11", "2026-12"]},
        {"id": "NovDec_lowcorr", "months": ["2026-11", "2026-12"],
         "overrides": {"corr": {"TTF_M/TTF_M1": 0.92}, "vol": {"HH_M1": 0.40}}},
        {"id": "DecJan_cancel", "months": ["2026-12", "2027-01"], "strike_usd_mmbtu": [22.0, 21.0],
         "can_cancel": True, "volume_mmbtu": 3_400_000},
        {"id": "MarApr", "months": ["2027-03", "2027-04"]},       # not calibrated
    ],
}


def test_many_trades_price_from_files(tmp_path):
    snap = synthetic_calibration(tmp_path / "2026-09-23")
    assert snapshot.resolve(tmp_path, "latest") == tmp_path / "2026-09-23"
    trades = deal.parse_trades(TRADES, PRICING)
    res = price_job.run(MARKET, trades, snap, AS_OF)

    tr = res["trades"].set_index("trade")
    assert list(tr.index) == ["NovDec", "NovDec_lowcorr", "DecJan_cancel"]
    assert "MarApr" in res["errors"] and "2027-03" in res["errors"]["MarApr"]

    # Lower TTF M/M+1 correlation -> more switching value.
    assert tr.loc["NovDec_lowcorr", "extrinsic_mc"] > tr.loc["NovDec", "extrinsic_mc"]
    assert np.isnan(tr.loc["DecJan_cancel", "value_kirk"])          # MC only with a floor
    assert tr.loc["NovDec", "value_kirk"] == pytest.approx(tr.loc["NovDec", "value_mc"], abs=0.02)

    legs = res["legs"].set_index(["trade", "leg"])
    assert legs.loc[("NovDec_lowcorr", "TTF_M1"), "corr_used_M_M1"] == pytest.approx(0.92)
    assert legs.loc[("NovDec_lowcorr", "HH_M1"), "vol"] == pytest.approx(0.40)
    # Samuelson: M+1 over the horizon is below its full-life implied; M is ~unchanged.
    assert legs.loc[("NovDec", "TTF_M1"), "vol"] < legs.loc[("NovDec", "TTF_M1"), "vol_file"]
    assert legs.loc[("NovDec", "TTF_M"), "vol"] == pytest.approx(
        legs.loc[("NovDec", "TTF_M"), "vol_file"], rel=0.02)
    # Must-deliver: with vol, the TTF deltas sum to a bit less than the cargo -
    # you end up delivering against whichever month's TTF fell relative to the
    # other. With zero vol they are exactly the cargo, discounted.
    cargo_mwh = -3_600_000 / 3.412141633
    ttf_mwh = legs.loc["NovDec"].loc[["TTF_M", "TTF_M1"], "qty_native_mc"].sum()
    assert 0.9 * cargo_mwh > ttf_mwh > cargo_mwh
    flat = deal.parse_trades({**TRADES, "trades": [
        {"id": "flat", "months": ["2026-11", "2026-12"], "overrides": {"vol_scale": 0.0}}]}, PRICING)[0]
    d = deal.build(flat, snap, HUB_SPECS, 0.04, AS_OF)
    row, flat_legs = price_job.price_deal(d, PRICING)
    df_chosen = d.payoff.df1 if row["prob_m1_mc"] == 1.0 else d.payoff.df0
    assert flat_legs["qty_native_mc"][flat_legs.hub == "TTF"].sum() == pytest.approx(cargo_mwh * df_chosen)

    port = price_job.portfolio(res["legs"])
    assert port["lots_mc"].sum() == pytest.approx(res["legs"]["lots_mc"].sum())
    assert set(port["future"]) == set(res["legs"]["future"])
    assert set(res["grids"]["trade"]) == set(tr.index)


def test_trade_parsing(tmp_path):
    trades = deal.parse_trades(TRADES, PRICING)
    assert (trades[0].k0, trades[0].k1) == (21.5, 21.5)
    assert (trades[2].k0, trades[2].k1) == (22.0, 21.0)
    assert trades[2].volume_mmbtu == 3_400_000 and trades[0].volume_mmbtu == 3_600_000

    dup = {**TRADES, "trades": TRADES["trades"][:1] * 2}
    with pytest.raises(ValueError, match="duplicate"):
        deal.parse_trades(dup, PRICING)

    snap = synthetic_calibration(tmp_path / "c")
    bad = deal.parse_trades({**TRADES, "trades": [
        {"id": "x", "months": ["2026-11", "2026-12"], "overrides": {"volatility": 1}}]}, PRICING)[0]
    with pytest.raises(ValueError, match="unknown override"):
        deal.build(bad, snap, HUB_SPECS, 0.04, AS_OF)


def test_edited_vol_in_the_file_flows_through(tmp_path):
    folder = tmp_path / "2026-09-23"
    synthetic_calibration(folder)
    c = pd.read_csv(folder / "contracts.csv")
    c.loc[(c.hub == "TTF") & (c.month == "2026-11"), "vol"] = 0.60
    edited = tmp_path / "2026-09-23_edited"
    folder.rename(edited)
    c.to_csv(edited / "contracts.csv", index=False)
    snap = snapshot.read(snapshot.resolve(tmp_path, "latest"))
    t = deal.parse_trades(TRADES, {**PRICING, "vol_horizon": "implied"})[0]
    d = deal.build(t, snap, HUB_SPECS, 0.04, AS_OF)
    assert d.legs.loc["TTF_M", "vol"] == pytest.approx(0.60)


def test_missing_vols_are_filled_from_the_previous_month_and_flagged():
    from DeliveryOptionModel.src.deliveryoption.market_data import fill_missing_vols
    c = pd.DataFrame({"hub": ["HH"] * 4 + ["TTF"], "month": ["2027-03", "2027-04", "2027-05", "2027-06", "2027-05"],
                      "fwd": [2.9, 2.7, 2.8, 2.9, 53.0], "atm_vol": [0.53, 0.45, np.nan, np.nan, np.nan],
                      "note": [""] * 5})
    out = fill_missing_vols(c, "previous")
    assert list(out["vol"][:4]) == [0.53, 0.45, 0.45, 0.45]
    assert out["atm_vol"].isna().sum() == 3                   # the Bloomberg read is untouched
    assert "FILLED from 2027-04" in out.loc[2, "note"]
    assert np.isnan(out.loc[4, "vol"])                        # nothing earlier to fill from
    assert fill_missing_vols(c, "none")["vol"].isna().sum() == 3


# ---------------------------------------------------------------------------
# long-dated additions
# ---------------------------------------------------------------------------

def test_anchors_count_back_from_today_not_from_m():
    far = calibration.anchor_months(pd.Period("2031-11", "M"), AS_OF, 48, same_pair=False)
    near = calibration.anchor_months(pd.Period("2026-11", "M"), AS_OF, 48, same_pair=False)
    assert far == near
    assert far[-1] == pd.Period("2026-09", "M") and len(far) == 48
    same = calibration.anchor_months(pd.Period("2031-11", "M"), AS_OF, 48, same_pair=True)
    assert [m.month for m in same] == [11] * 4


def test_profile_vol_is_the_rms_of_the_curve():
    p = pd.DataFrame({"tau_lo_days": [0, 30], "tau_hi_days": [30, np.inf], "realised_vol": [1.0, 0.5]})
    from DeliveryOptionModel.src.deliveryoption.vols import profile_vol
    assert profile_vol(p, 30) == pytest.approx(1.0)
    assert profile_vol(p, 90) == pytest.approx(np.sqrt((30 * 1.0 + 60 * 0.25) / 90))


def test_samuelson_fill_scales_the_curve_to_the_market_and_flags_it():
    from DeliveryOptionModel.src.deliveryoption.market_data import fill_missing_vols
    from DeliveryOptionModel.src.deliveryoption.vols import profile_vol
    prof = PROFILE[PROFILE.hub == "TTF"]
    months = pd.period_range("2027-01", "2027-06", freq="M")
    fix = [(m - 1).end_time.normalize() for m in months]
    base = [profile_vol(prof, (f - AS_OF).days) for f in fix]
    c = pd.DataFrame({"hub": "TTF", "month": [str(m) for m in months], "fwd": 50.0,
                      "fixing": fix, "option_expiry": fix, "note": "",
                      "atm_vol": [1.2 * b for b in base[:4]] + [np.nan, np.nan]})
    out = fill_missing_vols(c, "samuelson", PROFILE, AS_OF, scale_months=3)
    assert out["vol"].iloc[4] == pytest.approx(1.2 * base[4])
    assert out["vol"].iloc[5] == pytest.approx(1.2 * base[5])
    assert "samuelson" in out["note"].iloc[5] and out["atm_vol"].isna().sum() == 2


def test_fx_forward_curve_interpolates_pips_in_time():
    from DeliveryOptionModel.src.deliveryoption.market_data import fx_forward_curve
    as_of = pd.Timestamp("2026-01-15")
    curve = fx_forward_curve(1.10, {12: 100.0, 24: 300.0}, [pd.Period("2027-01", "M"),
                             pd.Period("2027-07", "M"), pd.Period("2030-01", "M")], as_of, 10000)
    assert curve["2027-01"] == pytest.approx(1.11, abs=2e-4)          # 12 months out
    assert curve["2027-07"] == pytest.approx(1.12, abs=2e-4)          # halfway to 24M
    assert curve["2030-01"] == pytest.approx(1.13)                    # flat past the last tenor


def test_pricing_uses_the_per_month_fx_forward(tmp_path):
    folder = tmp_path / "2026-09-23"
    synthetic_calibration(folder)
    c = pd.read_csv(folder / "contracts.csv")
    c["fx"] = np.where(c.hub == "TTF", 1.20, np.nan)
    c.to_csv(folder / "contracts.csv", index=False)
    snap = snapshot.read(folder)
    d = deal.build(deal.parse_trades(TRADES, PRICING)[0], snap, HUB_SPECS, 0.04, AS_OF)
    assert d.legs.loc["TTF_M", "fx_to_usd"] == pytest.approx(1.20)
    assert d.legs.loc["HH_M", "fx_to_usd"] == 1.0


def test_option_value_is_positive_even_when_the_deal_is_under_water(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    # Strike low enough that both months lose money: the deal is negative.
    t = deal.parse_trades({**TRADES, "trades": [
        {"id": "wet", "months": ["2026-11", "2026-12"], "strike_usd_mmbtu": [-5.0, -5.0]}]}, PRICING)[0]
    row, legs = price_job.price_deal(deal.build(t, snap, HUB_SPECS, 0.04, AS_OF), PRICING)
    assert row["value_mc"] < 0 < row["option_value_mc"]
    assert row["option_value_mc"] == pytest.approx(row["option_intrinsic"] + row["option_time_value_mc"])
    assert row["option_time_value_mc"] == pytest.approx(row["extrinsic_mc"])
    assert row["option_value_kirk"] == pytest.approx(row["option_value_mc"], abs=5 * row["option_stderr_mc"] + 0.02)


def test_option_hedge_at_zero_vol_is_switching_m_into_m1(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    t = deal.parse_trades({**TRADES, "trades": [
        {"id": "flat", "months": ["2026-11", "2026-12"], "overrides": {"vol_scale": 0.0}}]}, PRICING)[0]
    d = deal.build(t, snap, HUB_SPECS, 0.04, AS_OF)
    row, legs = price_job.price_deal(d, PRICING)
    assert row["prob_m1_mc"] == 1.0                      # synthetic curve: M+1 is the better month
    expected = d.payoff.df1 * d.payoff.w1 - d.payoff.df0 * d.payoff.w0
    got = legs["option_lots_mc"] * legs["lot_size"] * legs["fx_to_usd"] / t.volume_mmbtu
    assert got.to_numpy() == pytest.approx(expected)


def test_trades_csv_has_a_units_row_under_the_header(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    res = price_job.run(MARKET, deal.parse_trades(TRADES, PRICING)[:2], snap, AS_OF)
    path = tmp_path / "trades.csv"
    price_job.write_trades_csv(path, res["trades"], PRICING["bump_corr"])

    units = pd.read_csv(path, nrows=1, keep_default_na=False).iloc[0]
    assert units["option_value_mc"] == "USD/MMBtu"
    assert units["option_value_usd_mc"] == "USD"
    assert units["prob_m1_mc"] == "probability 0-1"
    assert units["corr_sens_usd_mc TTF_M/TTF_M1"] == "USD per +0.01 corr"
    assert (units != "").all()                                   # every column has one

    data = pd.read_csv(path, skiprows=[1])
    assert len(data) == 2 and data["option_value_mc"].dtype == float
    assert data["option_value_mc"].to_numpy() == pytest.approx(res["trades"]["option_value_mc"].to_numpy())

    with pytest.raises(KeyError, match="no unit"):
        price_job.trade_units(["option_value_mc", "brand_new_column"], 0.01)


def test_trades_report_the_vols_and_correlations_actually_used(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    res = price_job.run(MARKET, deal.parse_trades(TRADES, PRICING)[:2], snap, AS_OF)
    tr = res["trades"].set_index("trade")
    vols = [c for c in tr.columns if c.startswith("vol ")]
    corrs = [c for c in tr.columns if c.startswith("corr ")]
    assert vols == ["vol TTF_M", "vol TTF_M1", "vol HH_M", "vol HH_M1"]
    assert corrs == ["corr TTF_M/TTF_M1", "corr TTF_M/HH_M", "corr TTF_M/HH_M1",
                     "corr TTF_M1/HH_M", "corr TTF_M1/HH_M1", "corr HH_M/HH_M1"]
    # Overrides show up as used, not as calibrated.
    assert tr.loc["NovDec_lowcorr", "corr TTF_M/TTF_M1"] == pytest.approx(0.92, abs=1e-6)
    assert tr.loc["NovDec", "corr TTF_M/TTF_M1"] == pytest.approx(0.985, abs=1e-6)
    assert tr.loc["NovDec_lowcorr", "vol HH_M1"] == pytest.approx(0.40)
    # Same numbers as legs.csv's final `vol` column.
    legs = res["legs"].set_index(["trade", "leg"])["vol"]
    for leg in ("TTF_M", "TTF_M1", "HH_M", "HH_M1"):
        assert tr.loc["NovDec", f"vol {leg}"] == pytest.approx(legs[("NovDec", leg)])
    units = dict(zip(tr.reset_index().columns, price_job.trade_units(tr.reset_index().columns, 0.01)))
    assert units["vol TTF_M"] == "annualised vol, 0.30 = 30%"
    assert units["corr HH_M/HH_M1"] == "correlation -1..1"
    assert units["corr_sens_usd_mc TTF_M/TTF_M1"] == "USD per +0.01 corr"


def test_vol_haircuts_are_per_hub_and_skip_absolute_overrides(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    trades = deal.parse_trades(TRADES, PRICING)[:2]          # NovDec, NovDec_lowcorr (HH_M1 vol = 0.40)
    full = price_job.run(MARKET, trades, snap, AS_OF)
    cut = price_job.run({**MARKET, "pricing": {**PRICING, "TTF_vol_haircut": 0.8}}, trades, snap, AS_OF)

    lf = full["legs"].set_index(["trade", "leg"])["vol"]
    lc = cut["legs"].set_index(["trade", "leg"])["vol"]
    for leg in ("TTF_M", "TTF_M1"):
        assert lc[("NovDec", leg)] == pytest.approx(0.8 * lf[("NovDec", leg)])
    for leg in ("HH_M", "HH_M1"):
        assert lc[("NovDec", leg)] == pytest.approx(lf[("NovDec", leg)])     # HH untouched
    cut_hh = price_job.run({**MARKET, "pricing": {**PRICING, "HH_vol_haircut": 0.5}}, trades, snap, AS_OF)
    lh = cut_hh["legs"].set_index(["trade", "leg"])["vol"]
    assert lh[("NovDec", "HH_M")] == pytest.approx(0.5 * lf[("NovDec", "HH_M")])
    assert lh[("NovDec_lowcorr", "HH_M1")] == pytest.approx(0.40)             # typed in: not haircut

    tr = cut["trades"].set_index("trade")
    assert (tr["TTF_vol_haircut"] == 0.8).all() and (tr["HH_vol_haircut"] == 1.0).all()
    assert (tr["TTF_corr_haircut"] == 1.0).all() and (tr["HH_corr_haircut"] == 1.0).all()
    assert tr.loc["NovDec", "option_time_value_mc"] < full["trades"].set_index("trade").loc["NovDec", "option_time_value_mc"]
    assert any("TTF vol haircut" in n for n in cut["notes"]["NovDec"])
    assert price_job.trade_units(["TTF_vol_haircut", "HH_corr_haircut"], 0.01) == [
        "multiplier on calibrated vols", "multiplier on calibrated M/M+1 corr"]

    with pytest.raises(ValueError, match="TTF_vol_haircut"):
        deal.build(trades[0], snap, HUB_SPECS, 0.04, AS_OF, vol_haircuts={"TTF": 0.0})
    with pytest.raises(KeyError, match="TTF_vol_haircut"):
        price_job.run({**MARKET, "pricing": {**PRICING, "vol_haircut": 0.8}}, trades, snap, AS_OF)


def test_corr_haircut_moves_the_calibrated_m_m1_correlation(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    trades = deal.parse_trades(TRADES, PRICING)[:2]          # NovDec_lowcorr overrides TTF corr to 0.92
    full = price_job.run(MARKET, trades, snap, AS_OF)["trades"].set_index("trade")
    lower = price_job.run({**MARKET, "pricing": {**PRICING, "TTF_corr_haircut": 0.98}},
                          trades, snap, AS_OF)["trades"].set_index("trade")
    higher = price_job.run({**MARKET, "pricing": {**PRICING, "TTF_corr_haircut": 1.01}},
                           trades, snap, AS_OF)["trades"].set_index("trade")

    assert lower.loc["NovDec", "corr TTF_M/TTF_M1"] == pytest.approx(0.985 * 0.98, abs=1e-6)
    assert lower.loc["NovDec", "corr HH_M/HH_M1"] == pytest.approx(0.94, abs=1e-6)       # HH untouched
    assert lower.loc["NovDec_lowcorr", "corr TTF_M/TTF_M1"] == pytest.approx(0.92, abs=1e-6)  # typed in
    # Lower M/M+1 correlation -> worth more; higher -> worth less.
    assert lower.loc["NovDec", "option_value_mc"] > full.loc["NovDec", "option_value_mc"]
    assert higher.loc["NovDec", "option_value_mc"] < full.loc["NovDec", "option_value_mc"]

    with pytest.raises(ValueError, match="beyond 1"):
        deal.build(trades[0], snap, HUB_SPECS, 0.04, AS_OF, corr_haircuts={"TTF": 1.02})


def test_cross_hub_correlation_risk_bumps_all_four_pairs_together(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    t = deal.parse_trades(TRADES, PRICING)[0]
    row, _ = price_job.price_deal(deal.build(t, snap, HUB_SPECS, 0.04, AS_OF), PRICING)
    sens = {k[len("corr_sens_usd_mc "):]: v for k, v in row.items() if k.startswith("corr_sens_usd_mc ")}
    assert set(sens) == {"TTF_M/TTF_M1", "HH_M/HH_M1", "TTF/HH all cross pairs"}
    # A parallel cross-hub move leaves the calendar spreads' co-movement alone,
    # so it is small next to the TTF M/M+1 correlation risk.
    assert abs(sens["TTF/HH all cross pairs"]) < 0.2 * abs(sens["TTF_M/TTF_M1"])


REQUESTED_SUMMARY = (
    "trade,m0,m1,volume_mmbtu,sell,buy,k0,k1,can_cancel,decision_date,vol TTF_M,vol TTF_M1,vol HH_M,"
    "vol HH_M1,corr TTF_M/TTF_M1,corr TTF_M/HH_M,corr HH_M/HH_M1,option_value_mc,option_intrinsic,"
    "option_time_value_mc,option_value_usd_mc,option_value_kirk,extrinsic_mc,extrinsic_usd_mc,"
    "value_usd_mc,prob_m1_mc,corr_sens_usd_mc TTF_M/TTF_M1,corr_sens_usd_mc HH_M/HH_M1,"
    "corr_sens_usd_mc TTF/HH all cross pairs").split(",") + [
    f"option_delta_lots_mc {leg}" for leg in ("TTF_M", "TTF_M1", "HH_M", "HH_M1")] + [
    f"vega_usd_mc {leg}" for leg in ("TTF_M", "TTF_M1", "HH_M", "HH_M1")]


def test_trades_summary_is_the_requested_subset_with_units(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    res = price_job.run(MARKET, deal.parse_trades(TRADES, PRICING)[:2], snap, AS_OF)
    summary = price_job.trades_summary(res["trades"])
    assert list(summary.columns) == REQUESTED_SUMMARY          # exactly the list asked for
    full = res["trades"][REQUESTED_SUMMARY]
    rounded = [c for c in REQUESTED_SUMMARY if c not in price_job.SUMMARY_TERMS]
    assert (summary[rounded] == full[rounded].round(3)).all().all()
    assert (summary[rounded] - full[rounded]).abs().max().max() <= 0.0005 + 1e-12
    assert summary[price_job.SUMMARY_TERMS].equals(full[price_job.SUMMARY_TERMS])
    assert not res["trades"]["option_value_mc"].equals(summary["option_value_mc"])  # trades.csv keeps precision

    path = tmp_path / "trades_summary.csv"
    price_job.write_trades_csv(path, summary, PRICING["bump_corr"])
    units = pd.read_csv(path, nrows=1, keep_default_na=False).iloc[0]
    assert units["option_value_usd_mc"] == "USD" and units["vol TTF_M"] == "annualised vol, 0.30 = 30%"
    back = pd.read_csv(path, skiprows=[1])
    assert list(back.columns) == REQUESTED_SUMMARY and len(back) == 2

    with pytest.raises(KeyError, match="corr HH_M/HH_M1"):
        price_job.trades_summary(res["trades"].drop(columns=["corr HH_M/HH_M1"]))


def test_ttf_hh_corr_haircut_scales_the_four_cross_pairs_only(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    trades = deal.parse_trades(TRADES, PRICING)[:1]                 # NovDec: cross pairs all 0.3
    cross = ["corr TTF_M/HH_M", "corr TTF_M/HH_M1", "corr TTF_M1/HH_M", "corr TTF_M1/HH_M1"]
    base = price_job.run(MARKET, trades, snap, AS_OF)["trades"].set_index("trade")
    res = price_job.run({**MARKET, "pricing": {**PRICING, "TTF_HH_corr_haircut": 0.5}}, trades, snap, AS_OF)
    tr = res["trades"].set_index("trade")

    for c in cross:
        assert tr.loc["NovDec", c] == pytest.approx(0.5 * base.loc["NovDec", c], abs=1e-6)
    for c in ("corr TTF_M/TTF_M1", "corr HH_M/HH_M1"):
        assert tr.loc["NovDec", c] == pytest.approx(base.loc["NovDec", c], abs=1e-9)
    assert tr.loc["NovDec", "TTF_HH_corr_haircut"] == 0.5
    assert base.loc["NovDec", "TTF_HH_corr_haircut"] == 1.0
    assert any("TTF/HH cross corr haircut" in n for n in res["notes"]["NovDec"])
    assert price_job.trade_units(["TTF_HH_corr_haircut", "TTF_corr_haircut"], 0.01) == [
        "multiplier on calibrated cross-hub corr (all 4 pairs)", "multiplier on calibrated M/M+1 corr"]
    # With all four cross pairs equal, scaling them together leaves the calendar
    # spreads' co-movement exactly as it was: no material change in value.
    assert tr.loc["NovDec", "option_value_mc"] == pytest.approx(
        base.loc["NovDec", "option_value_mc"], abs=3 * base.loc["NovDec", "option_stderr_mc"])

    # Absolute per-trade overrides on the cross pairs win over the haircut.
    four = {p: 0.25 for p in ("TTF_M/HH_M", "TTF_M/HH_M1", "TTF_M1/HH_M", "TTF_M1/HH_M1")}
    t_ov = deal.parse_trades({**TRADES, "trades": [
        {"id": "ov", "months": ["2026-11", "2026-12"], "overrides": {"corr": four}}]}, PRICING)[0]
    d = deal.build(t_ov, snap, HUB_SPECS, 0.04, AS_OF, cross_corr_haircuts={("TTF", "HH"): 0.5})
    names = list(d.model.names)
    for pair in four:
        a, b = pair.split("/")
        assert d.model.corr[names.index(a), names.index(b)] == pytest.approx(0.25, abs=1e-6)
    assert not any("repaired" in n for n in d.notes)

    with pytest.raises(ValueError, match="beyond 1"):
        deal.build(trades[0], snap, HUB_SPECS, 0.04, AS_OF, cross_corr_haircuts={("TTF", "HH"): 4.0})
    with pytest.raises(ValueError, match="inconsistent"):             # 0.3 x 3.3 = 0.99: in range, not PSD
        deal.build(trades[0], snap, HUB_SPECS, 0.04, AS_OF, cross_corr_haircuts={("TTF", "HH"): 3.3})
    with pytest.raises(ValueError, match="TTF_HH_corr_haircut"):
        deal.build(trades[0], snap, HUB_SPECS, 0.04, AS_OF, cross_corr_haircuts={("TTF", "HH"): 0.0})


# ---------------------------------------------------------------------------
# Brent as a sell leg
# ---------------------------------------------------------------------------

HUB_SPECS_3 = {**HUB_SPECS, "Brent": {"future_root": "CO", "option_root": "CO", "fx": None,
                                      "mult": 1.0, "contract_month_offset": 2}}
MARKET_3 = {**MARKET, "hubs": HUB_SPECS_3}
BRENT_TRADES = {
    "defaults": {"volume_mmbtu": 3_600_000, "buy": {"underlying": "TTF", "weight": 1.0},
                 "can_cancel": False, "decision_date": None, "pay_lag_days": 20},
    "trades": [
        {"id": "B_NovDec", "months": ["2026-11", "2026-12"],
         "sell": {"underlying": "Brent", "weight": 0.166, "constant": 1.5}},
        {"id": "B_DecJan", "months": ["2026-12", "2027-01"],
         "sell": {"underlying": "Brent", "weight": [0.166, 0.132], "constant": 1.5}},
    ],
}


def synthetic_calibration_3(folder):
    """TTF, HH and Brent; Brent for delivery m is contract m+2, fixing end of m."""
    from DeliveryOptionModel.src.deliveryoption.market_data import hub_contract
    synthetic_calibration(folder)                                   # writes TTF + HH first
    c = pd.read_csv(folder / "contracts.csv")
    rows = []
    for i, m in enumerate(pd.period_range("2026-11", "2027-01", freq="M")):
        fix = m.end_time.normalize() - pd.Timedelta(days=1)
        rows.append({"hub": "Brent", "month": str(m), "future": hub_contract(HUB_SPECS_3["Brent"], m),
                     "fwd": [80.0, 79.0, 78.0][i], "fixing": fix.date(), "lot_size": 1000.0,
                     "atm_vol": 0.35, "vol_source": "synthetic",
                     "option_expiry": (fix - pd.Timedelta(days=3)).date(), "note": "", "vol": 0.35})
    pd.concat([c, pd.DataFrame(rows)]).to_csv(folder / "contracts.csv", index=False)
    brent_profile = PROFILE[PROFILE.hub == "TTF"].assign(hub="Brent", realised_vol=[0.40, 0.33])
    pd.concat([PROFILE, brent_profile]).to_csv(folder / "vol_profile.csv", index=False)
    names = leg_names(["TTF", "HH", "Brent"])
    hub = {n: n.split("_")[0] for n in names}
    within = {"TTF": .985, "HH": .94, "Brent": .99}
    across = {frozenset(("TTF", "HH")): .3, frozenset(("TTF", "Brent")): .35,
              frozenset(("HH", "Brent")): .2}
    m = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                m[i, j] = within[hub[a]] if hub[a] == hub[b] else across[frozenset((hub[a], hub[b]))]
    assert np.linalg.eigvalsh(m).min() > 0
    corr = pd.DataFrame(m, names, names)
    for f in (folder / "correlations").glob("*.csv"):
        corr.to_csv(f)
    return snapshot.read(folder)


def test_brent_contract_is_the_prompt_during_the_delivery_month():
    from DeliveryOptionModel.src.deliveryoption.market_data import hub_contract
    brent = HUB_SPECS_3["Brent"]
    assert hub_contract(brent, pd.Period("2028-11", "M")) == "COF29 Comdty"     # expires end Nov-28
    assert hub_contract(brent, pd.Period("2028-11", "M"), "option_root") == "COF29 Comdty"
    assert hub_contract(HUB_SPECS["TTF"], pd.Period("2028-11", "M")) == "TZTX28 Comdty"


def test_flexible_legs_parse_weights_and_constants_per_month():
    t0, t1 = deal.parse_trades(BRENT_TRADES, PRICING)
    assert (t0.sell_hub, t0.sell_weight, t0.sell_const) == ("Brent", (0.166, 0.166), (1.5, 1.5))
    assert (t1.sell_weight, t1.buy_hub, t1.buy_weight) == ((0.166, 0.132), "TTF", (1.0, 1.0))
    assert (t1.k0, t1.k1) == (1.5, 1.5)
    # The old form still reads: hub / slope, and strike_usd_mmbtu as the sell constant.
    old = deal.parse_trades(TRADES, PRICING)[2]
    assert (old.sell_hub, old.sell_weight, old.sell_const) == ("HH", (1.0, 1.0), (22.0, 21.0))
    both = {**BRENT_TRADES, "trades": [{**BRENT_TRADES["trades"][0], "strike_usd_mmbtu": 2.0}]}
    with pytest.raises(ValueError, match="not both"):
        deal.parse_trades(both, PRICING)
    bad = {**BRENT_TRADES, "trades": [{**BRENT_TRADES["trades"][0],
                                       "sell": {"underlying": "Brent", "wieght": 0.1}}]}
    with pytest.raises(ValueError, match="unknown sell keys"):
        deal.parse_trades(bad, PRICING)
    assert price_job.leg_label("Brent", (0.166, 0.166), (1.5, 1.5)) == "0.166*Brent + 1.5"
    assert price_job.leg_label("Brent", (0.166, 0.132), (1.5, 1.5)) == \
        "0.166*Brent + 1.5 (M) | 0.132*Brent + 1.5 (M+1)"


def test_brent_deal_weights_and_pricing(tmp_path):
    snap = synthetic_calibration_3(tmp_path / "c")
    t_nd, t_dj = deal.parse_trades(BRENT_TRADES, PRICING)
    d = deal.build(t_dj, snap, HUB_SPECS_3, 0.04, AS_OF)
    names = list(d.model.names)
    assert names == ["TTF_M", "TTF_M1", "Brent_M", "Brent_M1"]
    usd = 1.1414 / 3.412141633
    assert d.payoff.w0[names.index("Brent_M")] == pytest.approx(0.166)
    assert d.payoff.w1[names.index("Brent_M1")] == pytest.approx(0.132)       # the year-end step
    assert d.payoff.w0[names.index("TTF_M")] == pytest.approx(-usd)
    assert (d.payoff.k0, d.payoff.k1) == (1.5, 1.5)
    assert d.legs.loc["Brent_M", "future"] == "COG27 Comdty"                    # Dec-26 cargo -> COG27
    # Decision at the TTF M fixing, before the Brent contract fixes: Brent is still a forward.
    assert d.decision_date == d.legs.loc["TTF_M", "fixing"] < d.legs.loc["Brent_M", "fixing"]

    res = price_job.run(MARKET_3, [t_nd, t_dj], snap, AS_OF)
    assert not res["errors"]
    tr = res["trades"].set_index("trade")
    assert (tr["option_value_mc"] >= 0).all()
    assert tr.loc["B_DecJan", "sell"] == "0.166*Brent + 1.5 (M) | 0.132*Brent + 1.5 (M+1)"
    assert {"Brent_vol_haircut", "Brent_corr_haircut", "TTF_Brent_corr_haircut"} <= set(tr.columns)
    assert list(price_job.trades_summary(res["trades"]).columns) == [
        *price_job.SUMMARY_TERMS, "vol TTF_M", "vol TTF_M1", "vol Brent_M", "vol Brent_M1",
        "corr TTF_M/TTF_M1", "corr TTF_M/Brent_M", "corr Brent_M/Brent_M1", *price_job.SUMMARY_VALUES,
        "corr_sens_usd_mc TTF_M/TTF_M1", "corr_sens_usd_mc Brent_M/Brent_M1",
        "corr_sens_usd_mc TTF/Brent all cross pairs",
        *[f"option_delta_lots_mc {leg}" for leg in ("TTF_M", "TTF_M1", "Brent_M", "Brent_M1")],
        *[f"vega_usd_mc {leg}" for leg in ("TTF_M", "TTF_M1", "Brent_M", "Brent_M1")]]
    # The Brent vol haircut applies to Brent only.
    cut = price_job.run({**MARKET_3, "pricing": {**PRICING, "Brent_vol_haircut": 0.5}}, [t_nd], snap, AS_OF)
    base_legs = res["legs"].set_index(["trade", "leg"])["vol"]
    cut_legs = cut["legs"].set_index(["trade", "leg"])["vol"]
    assert cut_legs[("B_NovDec", "Brent_M")] == pytest.approx(0.5 * base_legs[("B_NovDec", "Brent_M")])
    assert cut_legs[("B_NovDec", "TTF_M")] == pytest.approx(base_legs[("B_NovDec", "TTF_M")])


def test_mixed_hh_and_brent_run_summarises_both(tmp_path):
    snap = synthetic_calibration_3(tmp_path / "c")
    trades = deal.parse_trades(TRADES, PRICING)[:1] + deal.parse_trades(BRENT_TRADES, PRICING)[:1]
    res = price_job.run(MARKET_3, trades, snap, AS_OF)
    summary = price_job.trades_summary(res["trades"]).set_index("trade")
    assert np.isnan(summary.loc["NovDec", "vol Brent_M"]) and np.isnan(summary.loc["B_NovDec", "vol HH_M"])
    assert {"corr_sens_usd_mc TTF/HH all cross pairs",
            "corr_sens_usd_mc TTF/Brent all cross pairs"} <= set(summary.columns)


def test_missing_vol_profile_fails_loudly_instead_of_nan():
    empty = PROFILE.iloc[0:0]
    with pytest.raises(ValueError, match="vol profile"):
        horizon_vol(0.35, 58, 60, 30, empty)


def test_trades_carry_per_leg_deltas_and_vegas(tmp_path):
    snap = synthetic_calibration(tmp_path / "c")
    res = price_job.run(MARKET, deal.parse_trades(TRADES, PRICING)[:2], snap, AS_OF)
    tr = res["trades"].set_index("trade")
    legs = res["legs"].set_index(["trade", "leg"])
    for t in ("NovDec", "NovDec_lowcorr"):
        for leg in ("TTF_M", "TTF_M1", "HH_M", "HH_M1"):
            assert tr.loc[t, f"delta_lots_mc {leg}"] == pytest.approx(legs.loc[(t, leg), "lots_mc"])
            assert tr.loc[t, f"option_delta_lots_mc {leg}"] == pytest.approx(legs.loc[(t, leg), "option_lots_mc"])
            assert tr.loc[t, f"vega_usd_mc {leg}"] == pytest.approx(legs.loc[(t, leg), "vega_usd_per_pt_mc"])
    # Selling HH / buying TTF: the whole delivery is long HH and short TTF lots.
    assert tr.loc["NovDec", "delta_lots_mc HH_M"] + tr.loc["NovDec", "delta_lots_mc HH_M1"] > 0
    assert tr.loc["NovDec", "delta_lots_mc TTF_M"] + tr.loc["NovDec", "delta_lots_mc TTF_M1"] < 0
    # The option alone switches M into M+1: its deltas in each hub roughly net out.
    opt_ttf = tr.loc["NovDec", "option_delta_lots_mc TTF_M"] + tr.loc["NovDec", "option_delta_lots_mc TTF_M1"]
    deal_ttf = tr.loc["NovDec", "delta_lots_mc TTF_M"] + tr.loc["NovDec", "delta_lots_mc TTF_M1"]
    assert abs(opt_ttf) < 0.2 * abs(deal_ttf)

    units = price_job.trade_units(["delta_lots_mc TTF_M", "option_delta_lots_mc TTF_M", "vega_usd_mc TTF_M"],
                                  0.01, 0.01)
    assert units == ["futures lots, whole delivery (hedge = minus)",
                     "futures lots, option only (hedge = minus)", "USD per +1 vol pt"]
    summary = price_job.trades_summary(res["trades"])
    assert "option_delta_lots_mc TTF_M" in summary and "vega_usd_mc HH_M1" in summary
    assert "delta_lots_mc TTF_M" not in summary              # whole-delivery delta stays in trades.csv
