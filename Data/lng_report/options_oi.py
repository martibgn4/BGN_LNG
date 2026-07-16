"""Options open-interest analysis toolkit.

Standalone (matplotlib) analysis of Bloomberg option-chain open interest:
parse tickers, bucket strikes, and render back-to-back OI-by-strike charts,
net/total OI grids and put/call-ratio history. Not part of the daily HTML
report — driven by ``generate_options_oi_plots``.
"""

import re
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from xbbg import blp

from Data.lng_report.config import bbg_dict

# Bloomberg bdp field for per-contract option gamma. xbbg lowercases field
# names, so the returned column is "gamma". Change here if your terminal uses a
# different mnemonic (e.g. "gamma_mid" / "opt_gamma").
GAMMA_FIELD = "gamma"

MONTH_CODES = {
    "F": (1, "Jan"), "G": (2, "Feb"), "H": (3, "Mar"), "J": (4, "Apr"),
    "K": (5, "May"), "M": (6, "Jun"), "N": (7, "Jul"), "Q": (8, "Aug"),
    "U": (9, "Sep"), "V": (10, "Oct"), "X": (11, "Nov"), "Z": (12, "Dec"),
}

# Root -> (base decade, product label). Bloomberg year is a single digit, so
# you must supply the decade yourself. Adjust base_decade if you trade a curve
# that straddles a decade boundary (see note in parse_ticker).
TICKER_RE = re.compile(
    r"^(?P<root>[A-Z]+?)"  # product root, e.g. CO
    r"(?P<mon>[FGHJKMNQUVXZ])"  # month code
    r"(?P<yr>\d)"  # single year digit
    r"(?P<cp>[PC])"  # P or C
    r"\s+(?P<strike>[\d.]+)"  # strike
    r"\s+(?P<yellow>\w+)$"  # yellow key (Comdty)
)


def parse_ticker(ticker: str, base_decade: int = 2020) -> dict:
    """Break one Bloomberg option ticker into its components."""
    m = TICKER_RE.match(ticker.strip())
    if not m:
        raise ValueError(f"Could not parse ticker: {ticker!r}")
    g = m.groupdict()
    mon_num, mon_lbl = MONTH_CODES[g["mon"]]
    # Single-digit year: 6 -> 2026 under base_decade=2020. If your curve spans
    # e.g. 2029->2030 you handle the rollover here (bump decade when yr < prev).
    year = base_decade + int(g["yr"])
    return {
        "ticker": ticker,
        "root": g["root"],
        "month_code": g["mon"],
        "month_num": mon_num,
        "year": year,
        "tenor": f"{mon_lbl}-{str(year)[-2:]}",   # e.g. Sep-26
        "sort_key": year * 100 + mon_num,          # chronological ordering
        "type": g["cp"],
        "strike": float(g["strike"]),
    }


def build_oi_frame(oi_dict: dict, base_decade: int = 2020) -> pd.DataFrame:
    """Turn the raw {ticker: oi} dict into a tidy dataframe."""
    rows = []
    for tkr, oi in oi_dict.items():
        rec = parse_ticker(tkr := tkr, base_decade)  # noqa: F841 (keep name)
        rec["oi"] = oi
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sort_values(["sort_key", "strike"]).reset_index(drop=True)


def bucket_strike(strike: float, width: float = 0.5) -> float:
    """
    Floor a strike to the nearest `width` grid below it.
        82.3, 82.4  -> 82.0
        82.5, 82.7  -> 82.5
    The +1e-9 nudge guards against float artefacts, so 82.5/0.5 floors to
    165 not 164.9999. Rounded at the end to keep clean bucket labels.
    """
    return round(int((strike + 1e-9) / width) * width, 10)


def plot_oi_by_tenor(oi_dict: dict, spots=None,
                     ticker=None,
                     base_decade: int = 2020, cols: int = 2,
                     bucket_width: float = 0.5,
                     savepath: str | None = None,
                     max_strike=100000.0,
                     min_strike=0.0,
                     weight: str = "oi", gamma_dict: dict | None = None):
    """
    One horizontal back-to-back Call/Put OI-by-strike chart per tenor.

    spot : draw a spot / forward reference line so you can read which strikes
           are the put floor (below) and the call wall (above).

    weight="oi"    : raw open interest per strike (default).
    weight="gamma" : open interest scaled by each option's gamma; requires
                     `gamma_dict` = {option_ticker: gamma}. Bars then read as
                     gamma exposure rather than lot count.
    """
    if weight not in ("oi", "gamma"):
        raise ValueError("weight must be 'oi' or 'gamma'")
    if weight == "gamma" and gamma_dict is None:
        raise ValueError("weight='gamma' requires gamma_dict")

    df = build_oi_frame(oi_dict, base_decade)

    # Per-bar value: raw OI, or OI scaled by per-contract gamma. Options with a
    # missing/NaN gamma contribute 0.
    if weight == "gamma":
        gamma = pd.to_numeric(df["ticker"].map(gamma_dict), errors="coerce").fillna(0.0)
        df["value"] = df["oi"] * gamma
    else:
        df["value"] = df["oi"]

    is_gamma = weight == "gamma"
    call_lbl = "Call Γ·OI" if is_gamma else "Call OI"
    put_lbl = "Put Γ·OI" if is_gamma else "Put OI"

    plot_mult = 0.1 if ticker == "HH" else 1.0

    # Chronological tenor order — you want Sep before Oct before Dec, not
    # alphabetical, so the panels read down the curve.
    tenors = (df[["tenor", "sort_key"]].drop_duplicates()
              .sort_values("sort_key")["tenor"].tolist())

    n = len(tenors)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows),
                             squeeze=False)
    axes = axes.ravel()

    CALL_C, PUT_C, SPOT_C = "#1baf7a", "#e34948", "#4a3aa7"

    for ax, tenor, spot in zip(axes, tenors, spots):
        sub = df[df["tenor"] == tenor].copy()
        sub = sub[sub["strike"] < max_strike]  # Max strike to plot for clarity
        sub = sub[sub["strike"] > min_strike]  # Min strike to plot for clarity
        # Aggregate onto a fixed strike grid: floor each strike to the bucket
        # below it, then SUM the OI of every raw strike landing in that bucket.
        sub["bucket"] = sub["strike"].apply(lambda s: bucket_strike(s, bucket_width))

        # Union of buckets across both types, sorted ascending so the y-axis
        # reads low-to-high strike from bottom to top.
        strikes = sorted(sub["bucket"].unique())
        calls = (sub[sub["type"] == "C"].groupby("bucket")["value"].sum()
                 .reindex(strikes, fill_value=0))
        puts = (sub[sub["type"] == "P"].groupby("bucket")["value"].sum()
                .reindex(strikes, fill_value=0))

        y = strikes  # range(len(strikes))
        ax.barh([i + 0.2 * plot_mult for i in y], calls.values, height=0.4 * plot_mult,
                color=CALL_C, label=call_lbl)
        ax.barh([i - 0.2 * plot_mult for i in y], puts.values, height=0.4 * plot_mult,
                color=PUT_C, label=put_lbl)

        # Thin the y ticks so labels never collide, however dense the ladder.
        ax.yaxis.set_tick_params(labelsize=10)

        ax.set_title(tenor, fontsize=11, loc="left")
        ax.set_xlabel("Gamma-weighted OI" if is_gamma else "Open interest (lots)")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

        if spot is not None:
            ax.axhline(spot, color=SPOT_C, ls="--", lw=0.8, alpha=0.5,
                       label=f"Spot {spot:g}")

        ax.legend(fontsize=8, frameon=False, loc="upper right")

    # Blank any unused panels.
    for ax in axes[n:]:
        ax.set_visible(False)

    suptitle = (f"{ticker} Options gamma-weighted OI by strike, per tenor" if is_gamma
                else f"{ticker} Options open interest by strike, per tenor")
    fig.suptitle(suptitle, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig


def _bucketed_oi_frame(oi_dict: dict, base_decade: int, anchors: list | None):
    """Build the parsed OI frame and assign each strike to its nearest anchor.

    Shared by the raw-OI and gamma-weighted grid builders so both bucket
    strikes identically.
    """
    df = build_oi_frame(oi_dict, base_decade).copy()
    if anchors is None:
        raise ValueError("Anchors are needed!!")
    anc = clean_anchors(anchors)
    report_empty_anchors(df["strike"].values, anc)
    df["bucket"] = bucket_to_anchors(df["strike"].values, anc)
    return df


def _grid_matrix(df, mode: str, net_col: str, total_col: str):
    """Pivot a bucketed frame into a (strike x tenor) matrix.

    mode="net" sums `net_col` (already signed +call/-put); mode="total" sums
    the unsigned `total_col`. Returns (matrix, strikes, tenors).
    """
    # Chronological tenor order (Sep before Oct before Dec, never alphabetical).
    tenors = (df[["tenor", "sort_key"]].drop_duplicates()
              .sort_values("sort_key")["tenor"].tolist())
    # Strikes ascending, so low strikes at the bottom of the grid.
    strikes = sorted(df["bucket"].unique())

    s_idx = {s: i for i, s in enumerate(strikes)}
    t_idx = {t: j for j, t in enumerate(tenors)}
    mat = np.zeros((len(strikes), len(tenors)))

    if mode == "net":
        agg = df.groupby(["bucket", "tenor"])[net_col].sum()
    elif mode == "total":
        agg = df.groupby(["bucket", "tenor"])[total_col].sum()
    else:
        raise ValueError("mode must be 'net' or 'total'")

    for (b, t), v in agg.items():
        mat[s_idx[b], t_idx[t]] = v

    return mat, strikes, tenors


def build_grid(oi_dict: dict, mode: str = "net", base_decade: int = 2020,
               anchors: list | None = None):
    """
    Returns (matrix, strikes, tenors) where matrix[i, j] is the cell value
    for strike i (ascending) and tenor j (chronological).

    mode="net":   calls - puts   (signed)
    mode="total": calls + puts   (unsigned)

    anchors : optional list of strikes to aggregate around (nearest-anchor).
              Fine near ATM, coarse in the wings. When given, it OVERRIDES
              bucket_width. When None, falls back to fixed-width buckets.
    """
    df = _bucketed_oi_frame(oi_dict, base_decade, anchors)

    # Signed OI per row: +OI for calls, -OI for puts. Summing this per
    # (bucket, tenor) gives net tilt directly; abs-summing gives total.
    df["signed"] = np.where(df["type"] == "C", df["oi"], -df["oi"])

    return _grid_matrix(df, mode, net_col="signed", total_col="oi")


def build_grid_gamma_weighted(oi_dict: dict, gamma_dict: dict, mode: str = "net",
                              base_decade: int = 2020, anchors: list | None = None):
    """
    Same grid as :func:`build_grid`, but every option's open interest is scaled
    by its (per-contract) gamma, so cells measure gamma-exposure concentration
    rather than a raw lot count. This surfaces where option gamma actually piles
    up — typically tighter around the forward than the raw-OI picture.

    gamma_dict : {option_ticker: gamma} aligned to `oi_dict`'s tickers (e.g. the
                 Bloomberg GAMMA field). Tickers with a missing/NaN gamma
                 contribute 0.

    mode="net":   call (Γ·OI) - put (Γ·OI)   (signed gamma tilt)
    mode="total": sum of Γ·OI over calls+puts (total gamma wall)

    Note: calls and puts share the same gamma at a given strike/expiry, so the
    net view reflects the call/put OI imbalance scaled by gamma sensitivity.
    """
    df = _bucketed_oi_frame(oi_dict, base_decade, anchors)

    # Attach per-option gamma; anything Bloomberg didn't return drops out as 0.
    df["gamma"] = pd.to_numeric(df["ticker"].map(gamma_dict), errors="coerce").fillna(0.0)
    df["gamma_oi"] = df["oi"] * df["gamma"]
    df["signed_gamma"] = np.where(df["type"] == "C", df["gamma_oi"], -df["gamma_oi"])

    return _grid_matrix(df, mode, net_col="signed_gamma", total_col="gamma_oi")


def plot_grid(oi_dict: dict, underlying: str = None, mode: str = "net", spot: float | None = None,
              base_decade: int = 2020,
              anchors: list | None = None,
              annotate: bool = True, savepath: str | None = None,
              weight: str = "oi", gamma_dict: dict | None = None,
              center_zero: bool = False):
    """Heatmap of the options OI grid.

    weight="oi"    : raw open interest (default, unchanged behaviour).
    weight="gamma" : open interest scaled by each option's gamma; requires
                     `gamma_dict` = {option_ticker: gamma}.

    center_zero : force a diverging scale centred on 0 (and label cells as Δ).
                  Use for day-over-day *change* grids, where even a total-mode
                  value can be negative (OI fell). Default False keeps the
                  normal total-mode sequential scale.
    """
    if weight == "gamma":
        if gamma_dict is None:
            raise ValueError("weight='gamma' requires gamma_dict")
        mat, strikes, tenors = build_grid_gamma_weighted(oi_dict, gamma_dict, mode, base_decade, anchors)
    elif weight == "oi":
        mat, strikes, tenors = build_grid(oi_dict, mode, base_decade, anchors)
    else:
        raise ValueError("weight must be 'oi' or 'gamma'")

    is_gamma = weight == "gamma"

    # Thin strike labels so a dense ladder never smears (same guard as before).
    n_s = len(strikes)
    step = max(1, n_s // 30)
    ytick_idx = list(range(0, n_s, step))

    fig, ax = plt.subplots(figsize=(1.4 * len(tenors) + 3, 0.32 * n_s + 2))

    # Diverging when signed values are expected: net mode always, or any change
    # grid (center_zero). Symmetric limit so green/red intensities are comparable.
    diverging = (mode == "net") or center_zero
    if diverging:
        lim = np.abs(mat).max() or 1
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
        cmap = "RdYlGn"  # red=puts/sold (floor), green=calls/bought (wall)
    else:
        norm = Normalize(vmin=0, vmax=(mat.max() or 1))
        cmap = "viridis"

    metric = "Γ·OI" if is_gamma else "OI"
    if mode == "net":
        cbar_lbl = f"Net {metric} (calls − puts)"
    else:
        cbar_lbl = f"Total {metric} (calls + puts)"
    if center_zero:
        cbar_lbl = "Δ " + cbar_lbl

    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap=cmap, norm=norm)

    ax.set_xticks(range(len(tenors)))
    ax.set_xticklabels(tenors, rotation=0)
    ax.set_yticks(ytick_idx)
    ax.set_yticklabels([f"{strikes[i]:g}" for i in ytick_idx])
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Strike")
    grid_kind = "Γ-weighted OI" if is_gamma else "OI"
    if center_zero:
        grid_kind = "Δ " + grid_kind
    ax.set_title(f"{underlying} Options {grid_kind} grid — {mode} view", loc="left", fontsize=12)

    # Spot line: interpolate onto the strike index so it sits between buckets.
    if spot is not None:
        below = [i for i, s in enumerate(strikes) if s <= spot]
        above = [i for i, s in enumerate(strikes) if s >= spot]
        if below and above:
            lo, hi = below[-1], above[0]
            ypos = lo if strikes[hi] == strikes[lo] else \
                lo + (spot - strikes[lo]) / (strikes[hi] - strikes[lo])
            ax.axhline(ypos, color="#4a3aa7", ls="--", lw=1.4)
            ax.text(-0.9, ypos, f" spot {spot:g}",
                    color="#4a3aa7", va="center", fontsize=8)

    # Annotate cells so the grid is readable at a glance. Raw OI is whole lots;
    # gamma-weighted values can be small, so pick decimals from the peak cell.
    if annotate and n_s * len(tenors) <= 400:
        vmax_abs = np.abs(mat).max()
        if not is_gamma:
            cell_fmt = "{:.0f}"
        elif vmax_abs >= 100:
            cell_fmt = "{:.0f}"
        elif vmax_abs >= 1:
            cell_fmt = "{:.1f}"
        else:
            cell_fmt = "{:.2f}"
        for i in range(n_s):
            for j in range(len(tenors)):
                v = mat[i, j]
                if abs(v) < 1e-9:
                    continue
                # White text on dark/intense cells, dark on pale. Diverging
                # scales are palest near the centre, so gauge intensity off norm.
                inten = abs(norm(v) - 0.5) * 2 if diverging else 0.7
                tc = "white" if inten > 0.6 else "#222"
                ax.text(j, i, cell_fmt.format(v), ha="center", va="center",
                        fontsize=7, color=tc)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_lbl)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig


def clean_anchors(anchors, warn=True):
    """
    Defensive cleanup of a user-supplied anchor list: sort, dedupe, and
    surface anything that looks wrong rather than silently swallowing it.
    Returns the cleaned sorted list.
    """
    raw = list(anchors)
    arr = sorted(set(float(a) for a in raw))

    if warn:
        # Report dropped duplicates.
        if len(arr) != len(raw):
            dupes = sorted({a for a in raw if raw.count(a) > 1})
            print(f"[anchors] removed duplicate strikes: {dupes}")
        # Note on typo detection: when wings are coarse, a stray anchor like 3
        # in a [3,20,30,...] list is nearly indistinguishable from a legitimate
        # deep-wing anchor by spacing alone. So we DON'T guess here. Instead,
        # the real check is data-driven and lives in report_empty_anchors():
        # a genuine anchor catches some OI; a typo sits in a dead region and
        # catches nothing. Run that after bucketing to catch bad anchors.
    return arr


def bucket_to_anchors(strikes, anchors):
    """
    Vectorised nearest-anchor assignment.
    `strikes` : 1-D array-like of raw strikes.
    `anchors` : sorted 1-D array of anchor strikes (use clean_anchors first).
    Returns an array of the same length as `strikes`, each entry the chosen
    anchor. Ties (exactly between two anchors) resolve DOWN, matching the
    docstring, so behaviour is deterministic.
    """
    strikes = np.asarray(strikes, dtype=float)
    anchors = np.asarray(anchors, dtype=float)

    # Midpoints between consecutive anchors are the band boundaries.
    # searchsorted with 'right' on the midpoints maps each strike to an anchor
    # index; ties land on the lower anchor because a strike exactly on a
    # midpoint returns the left band.
    mids = (anchors[:-1] + anchors[1:]) / 2.0
    idx = np.searchsorted(mids, strikes, side="left")
    idx = np.clip(idx, 0, len(anchors) - 1)
    return anchors[idx]


def report_empty_anchors(strikes, anchors, warn=True):
    """
    Data-driven anchor check: after assigning strikes to anchors, report any
    anchor that caught ZERO strikes. A legitimate anchor near live strikes
    catches OI; a typo (a 3 where you meant 73) sits in a dead region and
    catches nothing. This is far more reliable than guessing from spacing.

    Returns the list of empty anchors.
    """
    assigned = bucket_to_anchors(strikes, anchors)
    used = set(assigned.tolist())
    empty = [a for a in anchors if a not in used]
    if warn and empty:
        print(f"[anchors] these anchors caught NO strikes (possible typos or "
              f"dead levels): {[f'{a:g}' for a in empty]}")
    return empty


def plot_put_call_ois(df, underlying, active_contract, savepath=None):
    """df contains a bbg dataframe with total put, call and contract Open Interests"""
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.columns = ["put_oi", "call_oi", "oi", "total_put_oi", "total_call_oi", "total_oi"]

    # Compute put/call ratio
    df["put_call_ratio"] = df["put_oi"] / df["call_oi"]
    df_cleaned = df.dropna(subset=["put_call_ratio"])

    df["total_put_call_ratio"] = df["total_put_oi"] / df["total_call_oi"]
    df_cleaned2 = df.dropna(subset=["total_put_call_ratio"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Panel 1: Put/Call Ratio
    axes[0].plot(df_cleaned.index, df_cleaned["put_call_ratio"], color="steelblue", linewidth=1.2, label=f"{active_contract} Contract")
    axes[0].plot(df_cleaned2.index, df_cleaned2["total_put_call_ratio"], color="red", linewidth=1.2, label="All Contracts")
    axes[0].axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Ratio = 1")
    axes[0].set_ylim([0.5, 2.0])
    axes[0].set_title(f"{underlying} Options – Put/Call Ratio (Open Interest)")
    axes[0].set_ylabel("Put/Call Ratio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Underlying Open Interest
    axes[1].fill_between(df.index, df["call_oi"], alpha=0.5, label="Call OI", color="green")
    axes[1].fill_between(df.index, df["put_oi"], alpha=0.5, label="Put OI", color="red")
    axes[1].plot(df.index, df["oi"], linestyle="--", linewidth=2.0, label="Contract OI", color="black")
    axes[1].set_title(f" {underlying} - {active_contract} Put & Call Open Interest")
    axes[1].set_ylabel("Open Interest (contracts)")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: TOTAL Underlying Open Interest
    axes[2].fill_between(df.index, df["total_call_oi"], alpha=0.5, label="Call OI", color="green")
    axes[2].fill_between(df.index, df["total_put_oi"], alpha=0.5, label="Put OI", color="red")
    axes[2].plot(df.index, df["total_oi"], linestyle="--", linewidth=2.0, label="TOTAL OI", color="black")
    axes[2].set_title(f" {underlying} TOTAL Put & Call Open Interest")
    axes[2].set_ylabel("Open Interest (contracts)")
    axes[2].set_xlabel("Date")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")


def fetch_two_day_oi(option_tickers, lookback_days=7):
    """Return (oi_today, oi_prev) OPEN_INT dicts for the two most recent trading
    days available, via a bdh window that skips weekends/holidays.

    OPEN_INT is archived historically (unlike option greeks), so this gives the
    genuine day-over-day change in open interest per option ticker.
    """
    end = datetime.today()
    start = end - timedelta(days=lookback_days)
    hist = blp.bdh(tickers=option_tickers, flds="OPEN_INT", start_date=start, end_date=end)
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.droplevel(1)  # drop field level -> ticker columns
    hist = hist.dropna(how="all")
    if hist.shape[0] < 2:
        raise ValueError("Need at least two trading days of OI history to compare day-over-day")
    oi_today = hist.iloc[-1].dropna().to_dict()
    oi_prev = hist.iloc[-2].dropna().to_dict()
    return oi_today, oi_prev


def oi_change_dict(oi_today, oi_prev):
    """Per-ticker OI change (today - prev). Tickers present on only one day
    treat the missing side as 0 (a newly listed or expiring strike)."""
    tickers = set(oi_today) | set(oi_prev)
    return {t: float(oi_today.get(t, 0.0)) - float(oi_prev.get(t, 0.0)) for t in tickers}


def generate_options_oi_plots(underlying_str="Brent", compare_prev_day=False):
    if underlying_str in ["Brent", "TTF", "HH", "WTI"]:
        u_code = {
            "Brent": "CO",
            "TTF": "FJS",
            "HH": "NG",
            "WTI": "CL"
        }[underlying_str]

        bucket_width = {
            "Brent": 1.0,
            "WTI": 1.0,
            "TTF": 0.5,
            "HH": 0.1
        }[underlying_str]

        fixed_strikes_anchors = {
            "Brent": [20, 30, 40, 50, 60, 65, 80, 85, 90, 95, 100, 110, 120, 150, 200],
            "WTI": [20, 30, 40, 50, 60, 65, 73, 75, 78, 80, 85, 90, 95, 100, 110, 120, 150, 200],
            "TTF": [5, 10, 15, 20, 25, 30, 55, 60, 70, 80, 90, 100],
            "HH": [0.1, 0.5, 1.0, 1.5, 2.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]
        }[underlying_str]

        sym_levels = {
            "Brent": [1.0, 2.0, 3.0, 5.0],
            "WTI": [1.0, 2.0, 3.0, 5.0],
            "TTF": [0.5, 1.0, 2.0],
            "HH": [0.1, 0.5, 1.0]
        }[underlying_str]

        min_strike, max_strike = {
            "Brent": (30, 200),
            "WTI": (30, 200),
            "TTF": (5, 100),
            "HH": (0.1, 20),
        }[underlying_str]

    else:
        raise ValueError(f"Underlying {underlying_str} is not supported for options OI analysis.")

    # Configuration
    month, year = blp.bds(
        tickers=[f"{u_code}1 Comdty"],
        flds=["fut_month_yr"]
    )["value"].values[0].split(" ")

    month = month.capitalize()
    year_code = year[-1] if underlying_str != "HH" else year
    bbg_month_code = bbg_dict[month]

    TICKERS = [f"{u_code}{bbg_month_code}{year_code} Comdty"]  # front-month options aggregate
    FIELDS = ["OPEN_INT_TOTAL_PUT", "OPEN_INT_TOTAL_CALL", "OPEN_INT",
              "AGGREGATE_PUT_OPEN_INT", "AGGREGATE_CALL_OPEN_INT", "FUT_AGGTE_OPEN_INT"]
    dates = pd.date_range(end=datetime.today(), periods=300)
    START_DATE = dates[0]
    END_DATE = dates[-1]

    # Pull historical data
    df = blp.bdh(
        tickers=TICKERS,
        flds=FIELDS,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    plot_put_call_ois(
        df,
        underlying=f"{underlying_str}",
        active_contract=f"{month}-{year}",
        savepath=f"C:\\Marti\\{underlying_str}_P_C_history.png"
    )

    underlyings = [f"{u_code}{i} Comdty" for i in [1, 2, 3, 4, 5, 6]]

    chain_raw = blp.bds(
        tickers=underlyings,
        flds=["OPT_CHAIN"],
    )
    # Result is a DataFrame with one column containing option tickers
    option_tickers = chain_raw.iloc[:, 0].dropna().tolist()
    spots = blp.bdp(tickers=underlyings, flds=["px_last"])["px_last"].dropna().values

    # Pull open interest and per-contract gamma together so we can weight OI by
    # gamma. (Adjust GAMMA_FIELD if your terminal names the greek differently.)
    df_options_oi = blp.bdp(
        tickers=option_tickers,
        flds=["open_int", GAMMA_FIELD]
    )
    oi_dict = df_options_oi.to_dict()["open_int"]
    gamma_dict = df_options_oi.to_dict().get(GAMMA_FIELD, {})

    fig = plot_oi_by_tenor(
        oi_dict,
        ticker=underlying_str,
        bucket_width=bucket_width,
        savepath=f"C:\\Marti\\{underlying_str}_OI_by_tenor.png",
        spots=spots,
        max_strike=max_strike, min_strike=min_strike,
    )
    fig_gamma = plot_oi_by_tenor(
        oi_dict,
        ticker=underlying_str,
        bucket_width=bucket_width,
        savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_by_tenor.png",
        spots=spots,
        max_strike=max_strike, min_strike=min_strike,
        weight="gamma", gamma_dict=gamma_dict,
    )

    spot = spots[0]
    spot_int = int(spot)
    anchors = [spot_int + i for i in sym_levels] + [spot_int - i for i in sym_levels] + [spot_int] + fixed_strikes_anchors
    anchors = sorted(anchors)

    for mode in ["net", "total"]:
        # Raw open-interest grid.
        _ = plot_grid(
            oi_dict,
            savepath=f"C:\\Marti\\{underlying_str}_OI_options_{mode}.png",
            underlying=underlying_str,
            mode=mode,
            spot=spot,
            anchors=anchors,
        )
        # Gamma-weighted grid.
        _ = plot_grid(
            oi_dict,
            savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_options_{mode}.png",
            underlying=underlying_str,
            mode=mode,
            spot=spot,
            anchors=anchors,
            weight="gamma",
            gamma_dict=gamma_dict,
        )

    if compare_prev_day:
        # Day-over-day: real OI change from bdh, gamma held at today's values so
        # the picture isolates the positioning shift (Γ·ΔOI). Strikes that only
        # existed yesterday have no today-gamma and drop out (contribute 0).
        oi_today_h, oi_prev_h = fetch_two_day_oi(option_tickers)
        oi_change = oi_change_dict(oi_today_h, oi_prev_h)

        for mode in ["net", "total"]:
            _ = plot_grid(
                oi_change,
                savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_change_{mode}.png",
                underlying=f"{underlying_str} Δ1d",
                mode=mode,
                spot=spot,
                anchors=anchors,
                weight="gamma",
                gamma_dict=gamma_dict,
                center_zero=True,
            )
        _ = plot_oi_by_tenor(
            oi_change,
            ticker=f"{underlying_str} Δ1d",
            bucket_width=bucket_width,
            savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_change_by_tenor.png",
            spots=spots,
            max_strike=max_strike, min_strike=min_strike,
            weight="gamma", gamma_dict=gamma_dict,
        )


if __name__ == "__main__":
    generate_options_oi_plots(underlying_str="WTI")