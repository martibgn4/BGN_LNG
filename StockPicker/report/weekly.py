"""
============================================================
report/weekly.py  --  Weekly picks report (dark HTML)
============================================================
Renders the five picks with full attribution: why each name was
chosen, which factor families drove it, what changed since last
week, sector and risk exposure, and the model's own recent
track record.

Styling follows the BGN LNG desk convention (GitHub dark) so this
sits alongside the existing daily reports.
============================================================
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPORTS_DIR, SECTOR_COL
from ..features.registry import FEATURE_FAMILIES, family_of

BG = "#0D1117"
FG = "#C9D1D9"
BORDER = "#30363D"
PANEL = "#161B22"
GREEN = "#3FB950"
RED = "#F85149"
BLUE = "#58A6FF"
AMBER = "#D29922"


def _colour(v: float, invert: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return FG
    good = (v < 0) if invert else (v > 0)
    return GREEN if good else RED


def _pct(v, dp: int = 2) -> str:
    return f"{v*100:.{dp}f}%" if v is not None and np.isfinite(v) else "&mdash;"


def _num(v, dp: int = 2) -> str:
    return f"{v:.{dp}f}" if v is not None and np.isfinite(v) else "&mdash;"


def family_scores(row: pd.Series) -> dict[str, float]:
    """Average standardised score per factor family for one name.

    This is the attribution the report is built around: it answers "was this
    picked for momentum, for revisions, or for being cheap" rather than
    presenting a single opaque number.
    """
    out = {}
    for fam, feats in FEATURE_FAMILIES.items():
        vals = [row.get(f"{f}_z") for f in feats if f"{f}_z" in row.index]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            out[fam] = float(np.mean(vals))
    return out


def _bar(value: float, lo: float = -2.0, hi: float = 2.0) -> str:
    """Small inline bar for a z-score, centred on zero."""
    if value is None or not np.isfinite(value):
        return ""
    v = max(lo, min(hi, value))
    pct = (v - lo) / (hi - lo) * 100
    mid = (0 - lo) / (hi - lo) * 100
    colour = GREEN if v >= 0 else RED
    left, width = (mid, pct - mid) if v >= 0 else (pct, mid - pct)
    return (
        f'<div style="position:relative;height:10px;background:{BG};'
        f'border:1px solid {BORDER};border-radius:2px;min-width:90px;">'
        f'<div style="position:absolute;left:{mid}%;top:0;width:1px;height:100%;background:{BORDER};"></div>'
        f'<div style="position:absolute;left:{left}%;width:{max(width,0.8)}%;height:100%;background:{colour};'
        f'opacity:0.85;"></div></div>'
    )


def _table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["left"] * len(headers)
    th = "".join(
        f'<th style="text-align:{a};padding:7px 10px;border-bottom:2px solid {BORDER};'
        f'font-size:11px;text-transform:uppercase;letter-spacing:0.5px;color:#8B949E;">{h}</th>'
        for h, a in zip(headers, aligns)
    )
    trs = []
    for r in rows:
        tds = "".join(
            f'<td style="text-align:{a};padding:7px 10px;border-bottom:1px solid {BORDER};">{c}</td>'
            for c, a in zip(r, aligns)
        )
        trs.append(f"<tr>{tds}</tr>")
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f"<thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"
    )


def _panel(title: str, body: str, subtitle: str = "") -> str:
    sub = (
        f'<div style="font-size:12px;color:#8B949E;margin-bottom:10px;">{subtitle}</div>'
        if subtitle else ""
    )
    return (
        f'<div style="background:{PANEL};border:1px solid {BORDER};border-radius:8px;'
        f'padding:16px;margin-bottom:16px;">'
        f'<div style="font-size:15px;font-weight:600;margin-bottom:4px;">{title}</div>'
        f"{sub}{body}</div>"
    )


def build_report(
    book: pd.DataFrame,
    asof: dt.date,
    previous_book: pd.DataFrame | None = None,
    performance: dict | None = None,
    track_record: pd.DataFrame | None = None,
    universe_size: int | None = None,
    notes: list[str] | None = None,
) -> str:
    """Render the weekly report to an HTML string."""
    asof_s = pd.Timestamp(asof).strftime("%d-%b-%Y")

    # ---------------- picks table ----------------------------------------
    prev_secs = set(previous_book["security"]) if previous_book is not None and not previous_book.empty else set()
    rows = []
    for _, r in book.iterrows():
        sec = r["security"]
        name = html.escape(str(r.get("NAME", sec)))
        ticker = sec.replace(" US Equity", "")
        is_new = sec not in prev_secs
        badge = (
            f'<span style="background:{BLUE};color:{BG};padding:1px 6px;border-radius:3px;'
            f'font-size:10px;font-weight:600;margin-left:6px;">NEW</span>' if is_new and prev_secs else ""
        )
        fam = family_scores(r)
        top = sorted(fam.items(), key=lambda kv: -abs(kv[1]))[:3]
        drivers = ", ".join(
            f'<span style="color:{GREEN if v>0 else RED};">{k} {v:+.2f}</span>' for k, v in top
        )
        rows.append([
            f'<b style="font-size:14px;">{ticker}</b>{badge}<br>'
            f'<span style="color:#8B949E;font-size:11px;">{name}</span>',
            html.escape(str(r.get(SECTOR_COL, "&mdash;"))),
            f'<b>{_pct(r.get("weight"), 1)}</b>',
            _num(r.get("score"), 3),
            _pct(r.get("vol_60"), 1),
            _num(r.get("beta"), 2),
            drivers,
        ])
    picks = _table(
        ["Name", "Sector", "Weight", "Score", "Vol 60d", "Beta", "Top factor drivers"],
        rows,
        ["left", "left", "right", "right", "right", "right", "left"],
    )

    cash = float(book["cash_weight"].iloc[0]) if "cash_weight" in book.columns and len(book) else 0.0
    vol_scalar = float(book["vol_scalar"].iloc[0]) if "vol_scalar" in book.columns and len(book) else 1.0
    invested = float(book["weight"].sum()) if len(book) else 0.0

    exposure = _table(
        ["Metric", "Value"],
        [
            ["Invested", f'<b>{_pct(invested,1)}</b>'],
            ["Cash", f'<span style="color:{AMBER};">{_pct(cash,1)}</span>'],
            ["Vol-target scalar", _num(vol_scalar)],
            ["Positions", str(len(book))],
            ["Universe screened", str(universe_size) if universe_size else "&mdash;"],
            ["Weighted beta",
             _num(float((book["weight"] * pd.to_numeric(book.get("beta"), errors="coerce")).sum() / max(invested, 1e-9)))],
            ["Weighted vol 60d",
             _pct(float((book["weight"] * pd.to_numeric(book.get("vol_60"), errors="coerce")).sum() / max(invested, 1e-9)), 1)],
        ],
        ["left", "right"],
    )

    # ---------------- changes vs last week --------------------------------
    changes_html = ""
    if previous_book is not None and not previous_book.empty:
        cur = set(book["security"])
        added = cur - prev_secs
        dropped = prev_secs - cur
        crows = []
        for s in sorted(added):
            r = book[book["security"] == s].iloc[0]
            crows.append([
                f'<span style="color:{GREEN};font-weight:600;">BUY</span>',
                s.replace(" US Equity", ""),
                html.escape(str(r.get("NAME", ""))),
                _pct(r.get("weight"), 1),
                _num(r.get("score"), 3),
            ])
        for s in sorted(dropped):
            r = previous_book[previous_book["security"] == s].iloc[0]
            realised = r.get("fwd_ret_1w")
            crows.append([
                f'<span style="color:{RED};font-weight:600;">SELL</span>',
                s.replace(" US Equity", ""),
                html.escape(str(r.get("NAME", ""))),
                _pct(r.get("weight"), 1),
                f'<span style="color:{_colour(realised)};">{_pct(realised)}</span>' if realised is not None else "&mdash;",
            ])
        if crows:
            changes_html = _panel(
                "Changes since last rebalance",
                _table(["Action", "Ticker", "Name", "Weight", "Score / realised"], crows,
                       ["left", "left", "left", "right", "right"]),
                f"{len(added)} in, {len(dropped)} out &mdash; turnover drives cost, so a quiet week is a good week.",
            )
        else:
            changes_html = _panel(
                "Changes since last rebalance",
                '<div style="color:#8B949E;">No changes &mdash; all five names retained.</div>',
            )

    # ---------------- factor attribution grid -----------------------------
    fam_rows = []
    all_fams = sorted({f for _, r in book.iterrows() for f in family_scores(r)})
    for _, r in book.iterrows():
        fam = family_scores(r)
        cells = [f'<b>{r["security"].replace(" US Equity","")}</b>']
        for f in all_fams:
            v = fam.get(f)
            cells.append(
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<span style="color:{_colour(v)};min-width:38px;font-size:12px;">{_num(v)}</span>'
                f"{_bar(v)}</div>" if v is not None else "&mdash;"
            )
        fam_rows.append(cells)
    attribution = _panel(
        "Factor attribution",
        _table(["Ticker", *[f.title() for f in all_fams]], fam_rows,
               ["left"] + ["left"] * len(all_fams)),
        "Average standardised score per factor family. Positive = better than the sector-adjusted "
        "cross-section on that family. This is why each name is in the book.",
    )

    # ---------------- performance -----------------------------------------
    perf_html = ""
    if performance:
        p = performance
        prows = [
            ["CAGR (net)", f'<span style="color:{_colour(p.get("cagr"))};">{_pct(p.get("cagr"))}</span>'],
            ["Volatility", _pct(p.get("ann_vol"))],
            ["Sharpe", f'<b>{_num(p.get("sharpe"))}</b> <span style="color:#8B949E;">(t={_num(p.get("sharpe_t_stat"))})</span>'],
            ["Max drawdown", f'<span style="color:{RED};">{_pct(p.get("max_drawdown"))}</span>'],
            ["Benchmark CAGR", _pct(p.get("bench_cagr"))],
            ["Excess CAGR", f'<span style="color:{_colour(p.get("excess_cagr"))};">{_pct(p.get("excess_cagr"))}</span>'],
            ["Information ratio", _num(p.get("information_ratio"))],
            ["Alpha (ann)", f'{_pct(p.get("alpha_ann"))} <span style="color:#8B949E;">(t={_num(p.get("alpha_t_stat"))})</span>'],
            ["Up / down capture", f'{_num(p.get("up_capture"))} / {_num(p.get("down_capture"))}'],
        ]
        perf_html = _panel(
            "Backtested performance",
            _table(["Metric", "Value"], prows, ["left", "right"]),
            f'Walk-forward, survivorship-bias free, net of {p.get("cost_bps", 8)}bps round-trip costs. '
            "Past simulated performance is not a forecast.",
        )

    # ---------------- track record ----------------------------------------
    tr_html = ""
    if track_record is not None and not track_record.empty:
        trows = []
        for _, r in track_record.tail(12).iloc[::-1].iterrows():
            ex = r.get("excess")
            trows.append([
                pd.Timestamp(r["date"]).strftime("%d-%b-%y"),
                f'<span style="color:{_colour(r.get("strategy"))};">{_pct(r.get("strategy"))}</span>',
                _pct(r.get("benchmark")),
                f'<span style="color:{_colour(ex)};">{_pct(ex)}</span>',
                html.escape(str(r.get("names", ""))),
            ])
        tr_html = _panel(
            "Recent weekly track record",
            _table(["Week", "Strategy", "Benchmark", "Excess", "Holdings"], trows,
                   ["left", "right", "right", "right", "left"]),
            "Realised out-of-sample results, most recent first.",
        )

    notes_html = ""
    if notes:
        items = "".join(f"<li style='margin-bottom:5px;'>{html.escape(n)}</li>" for n in notes)
        notes_html = _panel("Notes and caveats", f"<ul style='margin:0;padding-left:18px;font-size:13px;color:#8B949E;'>{items}</ul>")

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>US Equity Weekly Picks &mdash; {asof_s}</title>
<style>
  body {{ background:{BG}; color:{FG}; font-family:-apple-system,Segoe UI,sans-serif;
          margin:0; padding:24px; max-width:1180px; margin-left:auto; margin-right:auto; }}
  h1 {{ font-size:22px; margin:0 0 4px 0; }}
  a {{ color:{BLUE}; }}
</style></head>
<body>
  <h1>US Equity &mdash; Weekly Picks</h1>
  <div style="color:#8B949E;font-size:13px;margin-bottom:20px;">
    Rebalance {asof_s} &nbsp;&middot;&nbsp; S&amp;P 500 universe &nbsp;&middot;&nbsp;
    long-only, inverse-vol weighted &nbsp;&middot;&nbsp; generated {dt.datetime.now():%d-%b-%Y %H:%M}
  </div>
  {_panel("This week's book", picks, "Ranked by walk-forward composite score. Weights are inverse-volatility, capped, and scaled to the volatility target.")}
  {_panel("Exposure", exposure)}
  {changes_html}
  {attribution}
  {perf_html}
  {tr_html}
  {notes_html}
  <div style="color:#6E7681;font-size:11px;margin-top:24px;border-top:1px solid {BORDER};padding-top:12px;">
    Generated by StockPicker. Research output, not investment advice. Bloomberg data.
  </div>
</body></html>"""


def save_report(html_str: str, asof: dt.date, directory: Path | None = None) -> Path:
    directory = directory or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"weekly_picks_{pd.Timestamp(asof):%Y-%m-%d}.html"
    path.write_text(html_str, encoding="utf-8")
    return path
