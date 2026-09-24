"""Map a trade's commercial terms and a calibration onto the engine.

Margin for delivery in month m, USD/MMBtu:

    X_m = (sell.weight_m * P_sell,m + sell.constant_m)
        - (buy.weight_m  * P_buy,m  + buy.constant_m)

Weights and constants may differ between M and M+1 (e.g. a Brent slope that
changes at a year end). For Brent the weight is the oil slope: 0.166 x
USD/bbl gives USD/MMBtu.

Each hub's price is converted to USD per its own unit by its registry entry
(TTF: x EURUSD / 3.412 to USD/MMBtu; HH and Brent: USD as quoted). Buy TTF /
sell 1.15 HH + K, buy TTF / sell 16.6% Brent + 1.5, and the usual US export
(buy 1.15 HH + K, sell TTF) are all the same code.

FX is deterministic: each contract converts at the EURUSD outright forward for
its delivery month (contracts.csv `fx`; spot for calibrations written before
that column existed). FX vol only enters through a quanto term on the
converted legs, second order here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from DeliveryOptionModel.src.deliveryoption.calibration import leg_names
from DeliveryOptionModel.src.deliveryoption.pricer import BasketModel, SwitchPayoff, nearest_correlation
from DeliveryOptionModel.src.deliveryoption.snapshot import Snapshot
from DeliveryOptionModel.src.deliveryoption.vols import horizon_vol

DAYS_PER_YEAR = 365.0
TAGS = ("M", "M1")


@dataclass(frozen=True)
class Trade:
    id: str
    m0: pd.Period
    m1: pd.Period
    volume_mmbtu: float
    sell_hub: str
    sell_weight: tuple[float, float]          # (M, M+1): weight on the sell underlying
    sell_const: tuple[float, float]           # (M, M+1): USD/MMBtu added to the sell price
    buy_hub: str
    buy_weight: tuple[float, float]
    buy_const: tuple[float, float]
    can_cancel: bool
    decision_date: pd.Timestamp | None
    pay_lag_days: int
    vol_horizon: str                          # samuelson | implied
    overrides: dict = field(default_factory=dict)

    @property
    def hubs(self) -> list[str]:
        return list(dict.fromkeys([self.buy_hub, self.sell_hub]))

    @property
    def k0(self) -> float:
        """Net constant of the M margin: sell constant - buy constant."""
        return self.sell_const[0] - self.buy_const[0]

    @property
    def k1(self) -> float:
        return self.sell_const[1] - self.buy_const[1]


def _per_month(value, what: str, trade_id: str) -> tuple[float, float]:
    """A scalar applies to both months; [M, M+1] sets each."""
    if np.isscalar(value):
        return float(value), float(value)
    if len(value) != 2:
        raise ValueError(f"{trade_id}: {what} must be a number or [M, M+1], got {value}")
    return float(value[0]), float(value[1])


def _leg_terms(raw: dict, side: str, trade_id: str) -> tuple[str, tuple, tuple]:
    """(underlying, weight, constant) of a sell / buy leg.

    New form: {underlying: Brent, weight: 0.166, constant: 1.5}. The older
    {hub: HH, slope: 1.15} is still read, with constant 0.
    """
    unknown = set(raw) - {"underlying", "weight", "constant", "hub", "slope"}
    if unknown:
        raise ValueError(f"{trade_id}: unknown {side} keys {sorted(unknown)}")
    underlying = raw.get("underlying", raw.get("hub"))
    weight = raw.get("weight", raw.get("slope"))
    if underlying is None or weight is None:
        raise ValueError(f"{trade_id}: {side} needs an underlying and a weight")
    return (str(underlying), _per_month(weight, f"{side}.weight", trade_id),
            _per_month(raw.get("constant", 0.0), f"{side}.constant", trade_id))


def parse_trades(cfg: dict, pricing_cfg: dict) -> list[Trade]:
    """trades.yaml -> Trades, each trade overriding `defaults` key by key."""
    defaults = cfg.get("defaults", {})
    trades = []
    for raw in cfg["trades"]:
        t = {**defaults, **raw}
        m0, m1 = (pd.Period(m, "M") for m in t["months"])
        if m1 <= m0:
            raise ValueError(f"{t['id']}: months must be increasing, got {t['months']}")
        tid = str(t["id"])
        sell_hub, sell_w, sell_c = _leg_terms(t["sell"], "sell", tid)
        buy_hub, buy_w, buy_c = _leg_terms(t["buy"], "buy", tid)
        # Older files put the sell constant in strike_usd_mmbtu. Accept either,
        # never both, so a constant cannot be counted twice.
        if "strike_usd_mmbtu" in t:
            if "constant" in t["sell"]:
                raise ValueError(f"{tid}: give sell.constant or strike_usd_mmbtu, not both")
            sell_c = _per_month(t["strike_usd_mmbtu"], "strike_usd_mmbtu", tid)
        trades.append(Trade(
            id=tid, m0=m0, m1=m1, volume_mmbtu=float(t["volume_mmbtu"]),
            sell_hub=sell_hub, sell_weight=sell_w, sell_const=sell_c,
            buy_hub=buy_hub, buy_weight=buy_w, buy_const=buy_c,
            can_cancel=bool(t.get("can_cancel", False)),
            decision_date=pd.Timestamp(t["decision_date"]) if t.get("decision_date") else None,
            pay_lag_days=int(t["pay_lag_days"]),
            vol_horizon=t.get("vol_horizon", pricing_cfg["vol_horizon"]),
            overrides=t.get("overrides") or {},
        ))
    ids = [t.id for t in trades]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate trade ids in {ids}")
    return trades


@dataclass(frozen=True)
class Deal:
    trade: Trade
    model: BasketModel
    payoff: SwitchPayoff
    decision_date: pd.Timestamp
    legs: pd.DataFrame        # per leg: contract, fwd, fixing, vols, units, lots
    notes: list[str]


def build(trade: Trade, snap: Snapshot, hub_specs: dict, rate: float,
          as_of: pd.Timestamp, vol_haircuts: dict | None = None,
          corr_haircuts: dict | None = None,
          cross_corr_haircuts: dict | None = None) -> Deal:
    """Haircuts are multipliers on the calibration; a missing key means 1.0.

    vol_haircuts:        {hub: x} on that hub's vols (after the horizon
                         adjustment): 0.8 prices at 80% of the vol.
    corr_haircuts:       {hub: x} on that hub's M / M+1 correlation. Below 1
                         the months decorrelate and the option is worth MORE;
                         to be conservative on an owned option, use > 1.
    cross_corr_haircuts: {(hub_a, hub_b): x} on all four cross pairs between
                         the two hubs (a_M/b_M, a_M/b_M1, a_M1/b_M,
                         a_M1/b_M1), each keeping its own level. For a
                         sell-HH / buy-TTF margin, below 1 is worth slightly
                         MORE; the effect is small.
    A haircut that leaves the matrix inconsistent (not PSD) is an error rather
    than a silent repair of the other correlations. A trade's `vol_scale`
    override multiplies on top; absolute `vol` and `corr` overrides are taken
    as given.
    """
    vol_haircuts, corr_haircuts = vol_haircuts or {}, corr_haircuts or {}
    cross_corr_haircuts = cross_corr_haircuts or {}
    for kind, cuts in (("vol", vol_haircuts), ("corr", corr_haircuts)):
        for hub, h in cuts.items():
            if not h > 0:
                raise ValueError(f"{hub}_{kind}_haircut must be > 0, got {h}")
    for (a, b), h in cross_corr_haircuts.items():
        if not h > 0:
            raise ValueError(f"{a}_{b}_corr_haircut must be > 0, got {h}")
    names = leg_names(trade.hubs)
    rows = []
    for hub in trade.hubs:
        for tag, month in zip(TAGS, (trade.m0, trade.m1)):
            c = snap.contract(hub, month)
            spec = hub_specs[hub]
            # Per-month FX forward if the calibration has one, else spot.
            if not spec.get("fx"):
                fx = 1.0
            elif "fx" in c.index and pd.notna(c["fx"]):
                fx = float(c["fx"])
            else:
                fx = snap.fx[spec["fx"]]
            rows.append({"leg": f"{hub}_{tag}", "hub": hub, "month": str(month),
                         "future": c["future"], "fwd": float(c["fwd"]),
                         "fixing": pd.Timestamp(c["fixing"]),
                         "option_expiry": pd.Timestamp(c["option_expiry"])
                         if pd.notna(c["option_expiry"]) else pd.Timestamp(c["fixing"]),
                         "atm_vol": float(c["atm_vol"]), "vol_file": float(c["vol"]),
                         "lot_size": float(c["lot_size"]),
                         "usd_per_native": fx * spec["mult"], "fx_to_usd": fx})
    legs = pd.DataFrame(rows).set_index("leg").loc[names]

    t_d = trade.decision_date or legs.loc[[f"{h}_M" for h in trade.hubs], "fixing"].min()
    h_days = np.array([max((min(t_d, fix) - as_of).days, 0) for fix in legs["fixing"]])
    legs["horizon_y"] = h_days / DAYS_PER_YEAR

    notes = [f"{leg} vol {r.vol_file:.4f} is not the Bloomberg read ({r.atm_vol:.4f})"
             for leg, r in legs.iterrows()
             if not np.isclose(r.vol_file, r.atm_vol, atol=1e-9)]
    legs["vol_horizon"] = [_vol(trade, leg, legs.loc[leg], h, snap, as_of)
                           for leg, h in zip(names, h_days)]
    legs["vol"] = legs["vol_horizon"] * legs["hub"].map(lambda h: vol_haircuts.get(h, 1.0))
    for hub in trade.hubs:
        if vol_haircuts.get(hub, 1.0) != 1.0:
            notes.append(f"{hub} vol haircut x{vol_haircuts[hub]:g}")

    corr = snap.pair_corr(trade.m0, trade.m1)
    missing = [n for n in names if n not in corr.index]
    if missing:
        raise KeyError(f"{trade.id}: correlation file for {trade.m0}/{trade.m1} lacks {missing}")
    corr = corr.loc[names, names].copy()
    calibrated_corr = corr.copy()
    for hub in trade.hubs:
        h = corr_haircuts.get(hub, 1.0)
        if h == 1.0:
            continue
        a, b = f"{hub}_M", f"{hub}_M1"
        cut = corr.loc[a, b] * h
        if abs(cut) > 1.0:
            raise ValueError(f"{trade.id}: {hub}_corr_haircut {h:g} takes {a}/{b} "
                             f"{corr.loc[a, b]:.4f} to {cut:.4f}, beyond 1")
        notes.append(f"{hub} M/M+1 corr haircut x{h:g}: {corr.loc[a, b]:.4f} -> {cut:.4f}")
        corr.loc[a, b] = corr.loc[b, a] = cut
    for (ha, hb), h in cross_corr_haircuts.items():
        if h == 1.0 or ha not in trade.hubs or hb not in trade.hubs:
            continue
        for x in TAGS:
            for y in TAGS:
                a, b = f"{ha}_{x}", f"{hb}_{y}"
                cut = corr.loc[a, b] * h
                if abs(cut) > 1.0:
                    raise ValueError(f"{trade.id}: {ha}_{hb}_corr_haircut {h:g} takes {a}/{b} "
                                     f"{corr.loc[a, b]:.4f} to {cut:.4f}, beyond 1")
                corr.loc[a, b] = corr.loc[b, a] = cut
        notes.append(f"{ha}/{hb} cross corr haircut x{h:g} on all four pairs")
    _require_haircut_kept_psd(trade, calibrated_corr, corr)

    legs, corr = _apply_overrides(trade, legs, corr, notes)
    c = corr.to_numpy()
    repaired = nearest_correlation(c)
    if not np.allclose(repaired, c, atol=1e-6):
        notes.append(f"correlation repaired to PSD (max change {np.abs(repaired - c).max():.4f})")

    w = {tag: np.zeros(len(names)) for tag in TAGS}
    for tag in TAGS:
        i = TAGS.index(tag)
        w[tag][names.index(f"{trade.sell_hub}_{tag}")] += trade.sell_weight[i]
        w[tag][names.index(f"{trade.buy_hub}_{tag}")] -= trade.buy_weight[i]
        w[tag] *= legs["usd_per_native"].to_numpy()

    df = [np.exp(-rate * _pay_time(m, trade.pay_lag_days, as_of)) for m in (trade.m0, trade.m1)]
    model = BasketModel(names=tuple(names), fwd=legs["fwd"].to_numpy(float),
                        vol=legs["vol"].to_numpy(float), horizon=legs["horizon_y"].to_numpy(float),
                        corr=repaired)
    payoff = SwitchPayoff(w0=w["M"], w1=w["M1"], k0=trade.k0, k1=trade.k1,
                          df0=df[0], df1=df[1], can_cancel=trade.can_cancel)
    return Deal(trade, model, payoff, t_d, legs, notes)


PSD_TOL = 1e-10


def _require_haircut_kept_psd(trade: Trade, before: pd.DataFrame, after: pd.DataFrame):
    """Fail if the haircuts made a consistent calibrated matrix inconsistent.

    The nearest-correlation repair downstream would otherwise quietly move
    correlations nobody asked to move.
    """
    lo_before = np.linalg.eigvalsh(before.to_numpy()).min()
    lo_after = np.linalg.eigvalsh(after.to_numpy()).min()
    if lo_before >= -PSD_TOL and lo_after < -PSD_TOL:
        raise ValueError(f"{trade.id}: the correlation haircuts leave an inconsistent matrix "
                         f"(min eigenvalue {lo_after:.2e}); use a smaller haircut")


def _vol(trade: Trade, leg: str, row: pd.Series, h_days: int, snap: Snapshot,
         as_of: pd.Timestamp) -> float:
    if trade.vol_horizon == "implied":
        return row["vol_file"]
    if trade.vol_horizon != "samuelson":
        raise ValueError(f"{trade.id}: unknown vol_horizon {trade.vol_horizon!r}")
    profile = snap.vol_profile[snap.vol_profile["hub"] == row["hub"]]
    return horizon_vol(row["vol_file"], (row["option_expiry"] - as_of).days,
                       (row["fixing"] - as_of).days, h_days, profile)


def _apply_overrides(trade: Trade, legs: pd.DataFrame, corr: pd.DataFrame, notes: list):
    """Per-trade scenario knobs, applied on top of the calibration.

        fwd:       {TTF_M: 70.0}              absolute, native units
        vol_scale: 1.1                        multiplies every horizon vol
        vol:       {TTF_M1: 0.80}             absolute horizon vol, wins over scale
        corr:      {TTF_M/TTF_M1: 0.97}       absolute pairwise correlation
    """
    o = trade.overrides
    unknown = set(o) - {"fwd", "vol_scale", "vol", "corr"}
    if unknown:
        raise ValueError(f"{trade.id}: unknown override keys {sorted(unknown)}")
    for leg, v in (o.get("fwd") or {}).items():
        legs.loc[_leg(legs, leg, trade), "fwd"] = float(v)
        notes.append(f"fwd {leg} = {v}")
    if o.get("vol_scale") is not None:
        legs["vol"] *= float(o["vol_scale"])
        notes.append(f"vols x {o['vol_scale']}")
    for leg, v in (o.get("vol") or {}).items():
        legs.loc[_leg(legs, leg, trade), "vol"] = float(v)
        notes.append(f"vol {leg} = {v}")
    for pair, v in (o.get("corr") or {}).items():
        a, b = (_leg(legs, x.strip(), trade) for x in pair.split("/"))
        corr.loc[a, b] = corr.loc[b, a] = float(v)
        notes.append(f"corr {a}/{b} = {v}")
    return legs, corr


def _leg(legs: pd.DataFrame, leg: str, trade: Trade) -> str:
    if leg not in legs.index:
        raise KeyError(f"{trade.id}: override names leg {leg!r}; legs are {list(legs.index)}")
    return leg


def _pay_time(month: pd.Period, lag_days: int, as_of: pd.Timestamp) -> float:
    pay = month.end_time.normalize() + pd.Timedelta(days=lag_days)
    return max((pay - as_of).days, 0) / DAYS_PER_YEAR
