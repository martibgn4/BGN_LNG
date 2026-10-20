"""Markdown forecast-comparison report from pipeline results.

A report built from synthetic data carries a banner on every page and states which world it is.
Numbers are printed as measured; nothing here re-ranks, filters or rescues a result.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from typing import Any

from ttf_forecaster.pipelines.synthetic_e2e import WorldResult

SYNTHETIC_BANNER = (
    "> **SYNTHETIC DATA — ARCHITECTURE CHECK ONLY.** Every number below comes from generated "
    "data (`data/synthetic.py`). It verifies that the pipeline runs and that the evaluation harness "
    "behaves correctly in a world with no signal (`null`) and a world with a planted signal "
    "(`planted`). **It is not evidence about TTF and must not be quoted as forecast skill.**"
)

_EXPECT = {
    "null": "no model should pass criterion 1; any pass is a leak or a false positive",
    "planted": "the full model should find the planted storage signal and beat its price-only twin",
}


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _criteria_table(crit: dict[str, dict[str, Any]], which: str) -> list[str]:
    lines = [
        f"| model vs `{which}` | crit. 1 pass | fold win share | HLN-DM p | CW p (suppl.) "
        "| OOS R² all | OOS R² ex-crisis | crit. 2 pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, c in crit.items():
        c1, c2 = c["criterion_1"], c["criterion_2"]
        lines.append(
            f"| `{m}` | {'PASS' if c1['passed'] else 'fail'} | {_fmt(c1['fold_win_share'], 2)} "
            f"| {_fmt(c1['dm_p_one_sided'])} | {_fmt(c1.get('cw_p_supplementary'))} "
            f"| {_fmt(c2['oos_r2_all'])} | {_fmt(c2['oos_r2_ex_crisis'])} "
            f"| {'PASS' if c2['passed'] else 'fail'} |"
        )
    return lines


def _trading_table(r: WorldResult) -> list[str]:
    lines = ["| model @ cost | net P&L (EUR/MWh) | Sharpe | max DD / ann vol | hit rate "
             "| top-5% days P&L ÷ net P&L | turnover/day |", "|---|---|---|---|---|---|---|"]
    for k, s in r.trading.items():
        lines.append(
            f"| `{k}` | {_fmt(s['net_pnl'], 2)} | {_fmt(s['sharpe'], 2)} "
            f"| {_fmt(s['max_dd_over_ann_vol'], 2)} | {_fmt(s['hit_rate'], 3)} "
            f"| {_fmt(s['pnl_top5pct_share'], 2)} | {_fmt(s['turnover_per_day'], 3)} |"
        )
    return lines


def _group_table(frames: dict, key: str) -> list[str]:
    lines = [f"| model | {key} | n | OOS R² vs zero | MSE model | MSE zero |", "|---|---|---|---|---|---|"]
    regime_names = ("oversupplied", "normal", "tight", "crisis")  # models/regimes.DEFAULT_LABELS
    for m, df in frames.items():
        for _, row in df.iterrows():
            g = row.get(key)
            if key == "year" and g is not None and not pd.isna(g):
                g = int(g)
            elif key == "regime" and isinstance(g, (int, float)) and not pd.isna(g) \
                    and 0 <= int(g) < len(regime_names):
                g = regime_names[int(g)]  # numeric code -> name; names pass through unchanged
            lines.append(f"| `{m}` | {g} | {int(row.get('n', 0))} | {_fmt(row.get('oos_r2'))} "
                         f"| {_fmt(row.get('mse'), 6)} | {_fmt(row.get('mse_bench'), 6)} |")
    return lines


def _world_section(r: WorldResult) -> list[str]:
    po = r.price_only_test
    syn = r.world in _EXPECT  # synthetic worlds carry their own expectations and caveats
    out = [
        f"## World: `{r.world}`",
        "",
        *([f"Expected behaviour: {_EXPECT[r.world]}.", ""] if syn else []),
        f"Run id `{r.run_id}` · {r.n_origins} origins · {r.n_features} features after exclusions.",
        "",
        f"Feature PIT grades all A/B (`clean_pit`, derived from the features): **{r.clean_pit}**.",
        "",
        "### Headline: spec §8 criteria 1–2 on BOTH labels (A1 / D-032 F7)",
        "",
        "A pass needs criteria 1 and 2 to pass on the fixed-contract label r(t, t+h) **and** on the "
        "executable label r(t+1, t+1+h) (forecasts scored on the settlement after the signal).",
        "",
        "| model | headline pass (both labels) |",
        "|---|---|",
        *[f"| `{m}` | {'PASS' if ok else 'fail'} |" for m, ok in r.headline_pass.items()],
        "",
        "### Spec §8 criteria 1–2 vs the zero (forward-curve) benchmark — fixed-contract label r(t, t+h)",
        "",
        *_criteria_table(r.criteria, "zero"),
        "",
        "### Same criteria on the EXECUTABLE label r(t+1, t+1+h) vs zero (A1)",
        "",
        *(_criteria_table(r.criteria_exec, "zero") if r.criteria_exec else ["(not computed)"]),
        "",
        f"### Same criteria vs the best benchmark in the set (`{r.best_benchmark}`, lowest pooled "
        "OOS MSE among zero / rolling mean / AR on own known lags)",
        "",
        *(_criteria_table(r.criteria_vs_best, r.best_benchmark) if r.criteria_vs_best else
          ["Best benchmark is the zero benchmark itself; table above applies."]),
        "",
        *(["Caveat: the synthetic generator has no crisis, so the ex-crisis column only checks "
           "that the code path runs.", ""] if syn else
          ["Crisis window for criterion 2: 2021-07-01..2023-06-30 (evaluation label only).", ""]),
        "### OOS R² vs zero by calendar year",
        "",
        *_group_table(r.year_metrics, "year"),
        "",
        "### OOS R² vs zero by point-in-time rule regime",
        "",
        *(_group_table(r.regime_metrics, "regime") if r.regime_metrics else ["(not computed)"]),
        "",
        "### Criterion 3: full feature set vs its price-only twin (same estimator)",
        "",
        f"- HLN-DM p = {_fmt(po['dm'].p_value)} (statistic {_fmt(po['dm'].statistic, 2)}) "
        "on r(t, t+h)",
        f"- HLN-DM p on the executable label r(t+1, t+1+h) = "
        f"{_fmt(r.price_only_test_exec['dm'].p_value) if r.price_only_test_exec else 'n/a'}"
        f" (statistic {_fmt(r.price_only_test_exec['dm'].statistic, 2) if r.price_only_test_exec else 'n/a'})",
        f"- Clark–West p = {_fmt(po['cw'].p_value)} (supplementary; measures predictability, "
        "not exploitable skill — modelling_log D-020). Its size with these estimators (separately "
        "tuned ridges, ~40 extra columns) is unverified; a CW p < 0.05 in the NULL world is a known "
        "false positive.",
        "- A positive DM statistic means the full model is WORSE than its price-only twin.",
        "",
        "### Trading simulation (next-settlement fill, held contract, non-overlapping h-day "
        "trades, cost ASSUMPTIONS)",
        "",
        *_trading_table(r),
        "",
        *(["Caveat: in the planted world P&L and Sharpe scale with the chosen planted β; they say "
           "nothing about achievable magnitudes. The planted signal is NOT executable at a "
           "next-settlement fill (model-risk F7).", ""] if syn else []),
        "The P&L concentration column is (P&L of the best 5% of days) ÷ (total net P&L): above 1 "
        "means the rest of the days lost money; it is unbounded or NaN when net P&L is near 0.",
        "The executable re-score uses the forecasts of the fixed-contract fit (not re-fitted on the "
        "shifted label), a conservative check. Costs are ASSUMPTIONS (bid/ask not available).",
        "",
        "### Stage timings (s)",
        "",
        ", ".join(f"{k} {v:.1f}" for k, v in r.timings.items()),
        "",
    ]
    return out


REAL_BANNER = (
    "> **REAL DATA — RESEARCH RESULT, NOT A TRADING RECOMMENDATION.** Licensed Bloomberg data "
    "(TZT settlements, GIE AGSI via `CGIEEU*`, EURUSD), point-in-time per "
    "`docs/point_in_time_policy.md`; storage is grade B (lag-approximated, revisions not "
    "reproducible). The locked holdout (origins from 2025-10-01) is NOT evaluated. Read with "
    "`docs/limitations.md` and the success criteria in `docs/research_specification.md` §8."
)


def render_report(results: Sequence[WorldResult], *, synthetic: bool = True,
                  title: str | None = None, intro: str | None = None) -> str:
    """Render the comparison report as Markdown (synthetic banner by default)."""
    banner = SYNTHETIC_BANNER if synthetic else REAL_BANNER
    head = title or ("Forecast comparison report — synthetic end-to-end run" if synthetic
                     else "Forecast comparison report — real data")
    lines = [f"# {head}", "", banner, ""]
    lines += [
        intro or (
            "Pipeline exercised: synthetic vendor frames → `pipelines.ingest` (raw + quality "
            "checks) → Parquet normalised layer → read back → point-in-time feature matrix → T3 "
            "fixed-contract targets with `label_known_time` → purged expanding walk-forward "
            "(6-month outer folds) → metrics, DM/CW tests, §8 criteria → trading simulation → "
            "immutable run artefacts."),
        "",
    ]
    for r in results:
        lines += _world_section(r)
        lines += ["", banner, ""]
    return "\n".join(lines)


def write_report(results: Sequence[WorldResult], path: str | Path, *, synthetic: bool = True,
                 title: str | None = None, intro: str | None = None) -> Path:
    """Write the report; returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_report(results, synthetic=synthetic, title=title, intro=intro),
                 encoding="utf-8")
    return p
