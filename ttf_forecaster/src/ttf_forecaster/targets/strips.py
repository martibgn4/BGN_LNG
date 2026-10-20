"""Quarter / season strips from monthly settlements (``docs/target_definitions.md`` §1).

A strip price is the **delivery-hours-weighted** mean of its constituent monthly settlements. It is
used only when every constituent month settled on both dates of a return/change; otherwise the
listed Q/S contract is used (``construction = listed``) if it settled on both dates; otherwise NaN.

Selection of the strip held from ``t`` for ``h`` sessions mirrors the held monthly contract: the
nearest strip whose **first** month still trades on the exit session (last trade >= exit + R') and,
when ``held_respects_roll_rule``, whose first month is beyond the roll window (> R sessions after t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.futures import FuturesContract, hours_weights
from ttf_forecaster.contracts.rolls import SettlementLookup
from ttf_forecaster.targets.common import CurveContext

_MAX_STEPS = 16


def strip_price(lookup: SettlementLookup, contract: FuturesContract, day: pd.Timestamp,
                tz: str = "Europe/Amsterdam") -> float:
    """Hours-weighted strip of the constituent months on ``day``; NaN if any month is missing."""
    months = contract.constituent_months()
    prices = np.array([lookup.get(m.contract_id, day) for m in months], dtype=float)
    if not np.all(np.isfinite(prices)):
        return float("nan")
    return float(np.dot(hours_weights(months, tz), prices))


def leg_prices(ctx: CurveContext, contract: FuturesContract, t: pd.Timestamp,
               exit_day: pd.Timestamp) -> tuple[float, float, str]:
    """Prices of a Q/S/Y leg on ``t`` and ``exit_day`` with the construction used.

    Monthly legs are returned as listed. Returns (p_t, p_exit, construction).
    """
    if contract.tenor_type == "M":
        return (ctx.lookup.get(contract.contract_id, t),
                ctx.lookup.get(contract.contract_id, exit_day), "single_contract")
    s0, s1 = strip_price(ctx.lookup, contract, t, ctx.tz), strip_price(ctx.lookup, contract,
                                                                         exit_day, ctx.tz)
    if np.isfinite(s0) and np.isfinite(s1):
        return s0, s1, "strip"
    l0 = ctx.lookup.get(contract.contract_id, t)
    l1 = ctx.lookup.get(contract.contract_id, exit_day)
    if np.isfinite(l0) and np.isfinite(l1):
        return l0, l1, "listed"
    return float("nan"), float("nan"), "strip"


def _containing(hub: str, tenor: str, day: pd.Timestamp) -> FuturesContract:
    if tenor == "Q":
        return FuturesContract.quarter(hub, day.year, (day.month - 1) // 3 + 1)
    if tenor == "S":
        if 4 <= day.month <= 9:
            return FuturesContract.season(hub, day.year, "SUM")
        return FuturesContract.season(hub, day.year if day.month >= 10 else day.year - 1, "WIN")
    raise ValueError(f"strip tenor must be Q or S, got {tenor!r}")


def held_strip(ctx: CurveContext, t: pd.Timestamp, exit_day: pd.Timestamp, tenor: str,
               season: str | None = None) -> FuturesContract | None:
    """Nearest Q or S strip held from ``t`` to ``exit_day`` (optionally only SUM or WIN)."""
    chain, rule = ctx.chain, ctx.rule
    t_pos = ctx.calendar.position(t)
    exit_pos = ctx.calendar.position(exit_day)
    cand = _containing(ctx.hub, tenor, t)
    for _ in range(_MAX_STEPS):
        if season is None or cand.season_code == season:
            first = cand.constituent_months()[0].contract_id
            i = chain.index_of.get(first)
            if i is None:
                if cand.delivery_start > chain.last_trade[-1]:
                    return None
            else:
                ltd = chain.ltd_pos[i]
                ok = ltd >= exit_pos + rule.exit_buffer_days
                if rule.held_respects_roll_rule:
                    ok = ok and (ltd - t_pos > rule.roll_days)
                if ok:
                    return cand
        cand = cand.next()
    return None
