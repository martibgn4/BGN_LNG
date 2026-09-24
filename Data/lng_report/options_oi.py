"""Options open-interest analysis toolkit.

Standalone (matplotlib) analysis of Bloomberg option-chain open interest:
parse tickers, bucket strikes, and render back-to-back OI-by-strike charts,
net/total OI grids and put/call-ratio history. Not part of the daily HTML
report — driven by ``generate_options_oi_plots``.
"""

import re
from datetime import datetime

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

# Same idea for per-contract delta. Bloomberg returns puts as negative and may
# quote in percent (-42.0) or in units (-0.42) depending on the terminal's
# DFLT settings — `normalise_deltas` below handles both.
DELTA_FIELD = "delta"

# Delta ladder resolution: 5% steps from 0 to 1 -> 20 buckets.
DELTA_BUCKET_WIDTH = 0.05

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

    # HH strikes sit ~10x closer together than the oil ones, so its bars need to
    # be a tenth as tall to stay separated. Prefix match, because `ticker` also
    # carries qualifiers by this point ("HH Δ15d", "HH (2026-06-15)").
    plot_mult = 0.1 if str(ticker).startswith("HH") else 1.0

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


def _grid_matrix(df, mode: str, net_col: str, total_col: str,
                 rows: list | None = None):
    """Pivot a bucketed frame into a (strike x tenor) matrix.

    mode="net" sums `net_col` (already signed +call/-put); mode="total" sums
    the unsigned `total_col`. Returns (matrix, strikes, tenors).

    rows : optional fixed row ladder. When None the rows are whichever buckets
           the data actually populated; pass a list to keep empty rows in the
           grid (used by the delta ladder, which is a fixed 0->1 axis).
    """
    # Chronological tenor order (Sep before Oct before Dec, never alphabetical).
    tenors = (df[["tenor", "sort_key"]].drop_duplicates()
              .sort_values("sort_key")["tenor"].tolist())
    # Strikes ascending, so low strikes at the bottom of the grid.
    strikes = sorted(df["bucket"].unique()) if rows is None else list(rows)

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

    # Spot line: interpolate onto the strike index so it sits between buckets.
    ref_pos = None
    if spot is not None:
        below = [i for i, s in enumerate(strikes) if s <= spot]
        above = [i for i, s in enumerate(strikes) if s >= spot]
        if below and above:
            lo, hi = below[-1], above[0]
            ref_pos = lo if strikes[hi] == strikes[lo] else \
                lo + (spot - strikes[lo]) / (strikes[hi] - strikes[lo])

    return _render_grid(
        mat, [f"{s:g}" for s in strikes], tenors,
        underlying=underlying, mode=mode, is_gamma=(weight == "gamma"),
        center_zero=center_zero, ylabel="Strike", annotate=annotate,
        ref_pos=ref_pos, ref_lbl=f" spot {spot:g}" if spot is not None else None,
        savepath=savepath,
    )


def _render_grid(mat, row_labels, tenors, *, underlying, mode, is_gamma,
                 center_zero=False, ylabel="Strike", row_axis_lbl="grid",
                 annotate=True, ref_pos=None, ref_lbl=None, savepath=None):
    """Draw a (row x tenor) OI matrix as an annotated heatmap.

    Shared by the strike-ladder grid (:func:`plot_grid`) and the delta-ladder
    grid (:func:`plot_delta_grid`); the two differ only in what a row means.

    row_labels  : pre-formatted y labels, one per matrix row (bottom-up).
    row_axis_lbl: how the rows are described in the title.
    ref_pos     : optional row coordinate (float, may sit between rows) for a
                  dashed reference line — spot on a strike ladder, 50Δ on a
                  delta ladder.
    """
    # Thin row labels so a dense ladder never smears (same guard as before).
    n_s = len(row_labels)
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
    ax.set_yticklabels([row_labels[i] for i in ytick_idx])
    ax.set_xlabel("Tenor")
    ax.set_ylabel(ylabel)
    grid_kind = "Γ-weighted OI" if is_gamma else "OI"
    if center_zero:
        grid_kind = "Δ " + grid_kind
    ax.set_title(f"{underlying} Options {grid_kind} {row_axis_lbl} — {mode} view",
                 loc="left", fontsize=12)

    if ref_pos is not None:
        ax.axhline(ref_pos, color="#4a3aa7", ls="--", lw=1.4)
        if ref_lbl:
            ax.text(-0.9, ref_pos, ref_lbl, color="#4a3aa7", va="center", fontsize=8)

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
                # White text on dark/intense cells, dark on pale.
                tc = "white" if abs(norm(v)) < 0.3 else "#222"
                ax.text(j, i, cell_fmt.format(v), ha="center", va="center",
                        fontsize=7, color=tc)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(cbar_lbl)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=1000, bbox_inches="tight")
    return fig


def normalise_deltas(deltas):
    """Coerce a delta series to signed units in [-1, 1].

    Bloomberg quotes delta in percent on some terminals (-42.0) and in units on
    others (-0.42). Scaling is decided once for the whole series — off the peak
    absolute value — so a mixed rescale can never split one strike across two
    buckets. Non-numeric entries become NaN.
    """
    d = pd.to_numeric(pd.Series(deltas), errors="coerce")
    if d.abs().max() > 1.5:  # percent-quoted terminal
        d = d / 100.0
    return d


def to_delta_space(deltas, types):
    """Map each contract onto a common 0->1 axis that rises with strike.

    Puts use |Δ| directly, calls use 1 − |Δ|. By put/call parity these agree,
    so a call and a put on the same strike/expiry land in the same bucket and
    their OI aggregates together:

        0.05  deep OTM put   (low strikes, the put floor)
        0.50  at-the-money
        0.95  deep OTM call  (high strikes, the call wall)
    """
    d = np.abs(np.asarray(deltas, dtype=float))
    is_call = np.asarray(types) == "C"
    return np.where(is_call, 1.0 - d, d)


def bucket_delta(deltas, width: float = DELTA_BUCKET_WIDTH):
    """Floor delta-space values onto the fixed `width` ladder.

    The +1e-9 nudge matters here: `1 - 0.90` is 0.09999999999999998 in binary
    float, which would floor a 90Δ call into the 5-10Δ bucket while its
    same-strike 10Δ put lands on 10-15Δ — splitting one strike in two. Indices
    are clipped to the ladder, so a 0Δ/100Δ contract lands in the end bucket
    rather than falling off it.
    """
    n = int(round(1.0 / width))
    idx = np.floor((np.asarray(deltas, dtype=float) + 1e-9) / width)
    return np.round(np.clip(idx, 0, n - 1) * width, 10)


def delta_ladder(width: float = DELTA_BUCKET_WIDTH):
    """The fixed list of bucket left-edges from 0 to 1, ascending."""
    return [round(i * width, 10) for i in range(int(round(1.0 / width)))]


def delta_bucket_label(bucket: float, width: float = DELTA_BUCKET_WIDTH) -> str:
    """Human label for a bucket left-edge, e.g. 0.45 -> "45-50Δ"."""
    return f"{bucket * 100:.0f}-{(bucket + width) * 100:.0f}Δ"


def build_delta_grid(oi_dict: dict, delta_dict: dict, mode: str = "total",
                     base_decade: int = 2020, width: float = DELTA_BUCKET_WIDTH,
                     weight: str = "oi", gamma_dict: dict | None = None):
    """
    Returns (matrix, buckets, tenors) where matrix[i, j] is the aggregated OI
    for delta bucket i (ascending, fixed 0->1 ladder) and tenor j.

    Same aggregation as :func:`build_grid`, but rows are moneyness clusters
    rather than absolute strikes — so the picture is comparable across tenors
    and across days even as the forward moves.

    delta_dict : {option_ticker: delta} (e.g. the Bloomberg DELTA field).
                 Contracts Bloomberg returned no delta for are dropped, since
                 they cannot be placed on the ladder.

    mode="net":   calls - puts   (signed)
    mode="total": calls + puts   (unsigned)
    weight="gamma" additionally scales every OI by the contract's gamma.
    """
    if weight not in ("oi", "gamma"):
        raise ValueError("weight must be 'oi' or 'gamma'")
    if weight == "gamma" and gamma_dict is None:
        raise ValueError("weight='gamma' requires gamma_dict")

    df = build_oi_frame(oi_dict, base_decade).copy()
    df["delta"] = normalise_deltas(df["ticker"].map(delta_dict)).values

    missing = int(df["delta"].isna().sum())
    if missing:
        print(f"[delta] {missing} of {len(df)} contracts had no delta and were "
              f"excluded from the delta grid")
    df = df.dropna(subset=["delta"])
    if df.empty:
        raise ValueError("No contracts with a usable delta — check DELTA_FIELD")

    df["bucket"] = bucket_delta(to_delta_space(df["delta"], df["type"]), width)

    if weight == "gamma":
        gamma = pd.to_numeric(df["ticker"].map(gamma_dict), errors="coerce").fillna(0.0)
        df["value"] = df["oi"] * gamma
    else:
        df["value"] = df["oi"]
    df["signed"] = np.where(df["type"] == "C", df["value"], -df["value"])

    # Fixed ladder: keep empty wing buckets so the axis is identical every run.
    return _grid_matrix(df, mode, net_col="signed", total_col="value",
                        rows=delta_ladder(width))


def plot_delta_grid(oi_dict: dict, delta_dict: dict, underlying: str = None,
                    mode: str = "total", base_decade: int = 2020,
                    width: float = DELTA_BUCKET_WIDTH,
                    annotate: bool = True, savepath: str | None = None,
                    weight: str = "oi", gamma_dict: dict | None = None,
                    center_zero: bool = False):
    """Heatmap of aggregated OI per delta bucket (rows) and tenor (columns).

    The strike-grid twin of :func:`plot_grid`, in moneyness space: the bottom of
    the chart is the OTM put wing, the middle is ATM, the top is the OTM call
    wing. A dashed line marks the 50Δ boundary.
    """
    mat, buckets, tenors = build_delta_grid(
        oi_dict, delta_dict, mode, base_decade, width, weight, gamma_dict)

    # The ATM line sits on the boundary *between* the two buckets straddling
    # 0.50, i.e. half a row below the first bucket at/above it.
    atm_row = next((i for i, b in enumerate(buckets) if b >= 0.5), None)

    return _render_grid(
        mat, [delta_bucket_label(b, width) for b in buckets], tenors,
        underlying=underlying, mode=mode, is_gamma=(weight == "gamma"),
        center_zero=center_zero, ylabel="Delta bucket (put wing → call wing)",
        row_axis_lbl="by delta", annotate=annotate,
        ref_pos=None if atm_row is None else atm_row - 0.5,
        ref_lbl=" 50Δ / ATM", savepath=savepath,
    )


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

    # The rename below is positional, so a short or empty response would silently
    # mislabel the series. Catch it here rather than several frames deep in
    # matplotlib — an empty frame means the aggregate contract ticker is wrong.
    if df.empty or df.shape[1] != 6:
        raise ValueError(f"Expected 6 open-interest columns of history for "
                         f"{active_contract}, got {df.shape[1]} column(s) and "
                         f"{len(df)} row(s) — check the aggregate contract ticker")

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


# ---------------------------------------------------------------------------
# As-of-date plumbing
# ---------------------------------------------------------------------------
# Everything below exists because Bloomberg archives some of this data and not
# the rest. What a `bdh` call can actually reach for a past date:
#
#   archived      OPEN_INT, PX_LAST, PX_VOLUME — for live *and* expired
#                 contracts, so a past OI snapshot is genuine data.
#   NOT archived  DELTA / GAMMA — bdh returns an empty frame even for a live
#                 option, so historical runs have no greeks at all.
#   NOT archived  OPT_CHAIN — a live bulk field. CHAIN_DATE and
#                 SINGLE_DATE_OVERRIDE are both ignored (you get today's chain
#                 back), and an expired contract returns nothing. Hence the
#                 chain reconstruction in `fetch_option_chain`.
#   NOT archived  FUT_MONTH_YR on a generic — resolved via `blp.fut_ticker`.
#
# Convenient side effect: bdh silently drops tickers it doesn't recognise, so a
# reconstructed chain that overshoots the strikes that really listed is safe.

# A futures contract symbol, e.g. COV6 (one year digit) or COQ26 (two).
FUT_SYMBOL_RE = re.compile(r"^(?P<root>[A-Z]+)(?P<mon>[FGHJKMNQUVXZ])(?P<yr>\d{1,2})$")

# An option ticker split into the parts we rebuild it from.
CHAIN_TICKER_RE = re.compile(
    r"^(?P<sym>[A-Z]+[FGHJKMNQUVXZ]\d)(?P<cp>[PC])"
    r"\s+(?P<strike>[\d.]+)\s+(?P<yellow>\w+)$"
)

# Bloomberg left-justifies "<symbol><P|C>" to this width before the strike
# ("COV6P    85 Comdty", "FJSU6P   85 Comdty").
CHAIN_SYMBOL_WIDTH = 9

# Window ending on the as-of date that a single-day snapshot is taken from.
# Wide enough to ride over a long weekend or a contract that didn't print.
ASOF_LOOKBACK_DAYS = 10


def as_of_timestamp(_date):
    """Normalise a caller-supplied date to a midnight Timestamp, or None."""
    return None if _date is None else pd.Timestamp(_date).normalize()


def is_historical(as_of) -> bool:
    """True when `as_of` is a genuinely past date, so we must go via bdh.

    None (and today's date) mean "latest", which keeps the bds/bdp path and its
    live greeks.
    """
    return as_of is not None and as_of.date() < datetime.today().date()


def option_root(contract_ticker: str) -> str:
    """Futures contract -> the root its option tickers carry.

    Option chains always use a single year digit ('COQ6P    85 Comdty'), which
    is also all :func:`parse_ticker` accepts, so the two-digit form that
    `blp.fut_ticker` returns for past dates ('COQ26 Comdty') is squeezed down.
    """
    sym = contract_ticker.split()[0]
    m = FUT_SYMBOL_RE.match(sym)
    if not m:
        raise ValueError(f"Could not parse futures contract: {contract_ticker!r}")
    g = m.groupdict()
    return f"{g['root']}{g['mon']}{g['yr'][-1]}"


def full_contract_ticker(contract: str, as_of) -> str:
    """Normalise a futures contract to its two-digit-year form.

    `blp.fut_ticker` drops to a single year digit when the contract falls in the
    current month ('COV6 Comdty'), and OPT_CHAIN rejects that form for some
    products — NG answers to 'NGU26 Comdty' but not 'NGU6 Comdty', even though
    its *options* still carry a single digit. The two-digit form works for every
    product here, so everything asking Bloomberg about a contract goes through
    this; only option tickers are built off :func:`option_root`.
    """
    sym, *rest = contract.split()
    m = FUT_SYMBOL_RE.match(sym)
    if not m:
        raise ValueError(f"Could not parse futures contract: {contract!r}")
    g = m.groupdict()
    if len(g["yr"]) == 2:
        return contract
    # One digit is decade-ambiguous. The contract is never before the as-of
    # month (fut_ticker only returns ones still trading), which pins the decade.
    year = (as_of.year // 10) * 10 + int(g["yr"])
    if (year, MONTH_CODES[g["mon"]][0]) < (as_of.year, as_of.month):
        year += 10
    return " ".join([f"{g['root']}{g['mon']}{year % 100:02d}", *rest])


def chain_ladder(option_tickers):
    """The (call/put, strike-as-written, yellow key) ladder behind a chain.

    Keeps the strike exactly as Bloomberg wrote it ('1', '6.25', '.25') so a
    rebuilt ticker is character-identical to a real one.
    """
    ladder, seen = [], set()
    for tkr in option_tickers:
        m = CHAIN_TICKER_RE.match(tkr.strip())
        if not m:
            continue
        key = (m["cp"], m["strike"], m["yellow"])
        if key not in seen:
            seen.add(key)
            ladder.append(key)
    return ladder


def build_chain_tickers(root: str, ladder):
    """Rebuild the option chain for `root` off a ladder taken from a live one."""
    return [f"{root}{cp}".ljust(CHAIN_SYMBOL_WIDTH) + f"{strike} {yellow}"
            for cp, strike, yellow in ladder]


def fetch_front_contract(u_code: str, as_of=None):
    """('Aug', '26', contract) — the front contract on `as_of`.

    `contract` is the exact two-digit-year ticker ('COQ25 Comdty'), and is None
    on a live pull where the caller's own short-year convention still resolves.
    """
    contract = None
    if not is_historical(as_of):
        label = blp.bds(tickers=[f"{u_code}1 Comdty"],
                        flds=["fut_month_yr"])["value"].values[0]
    else:
        # FUT_MONTH_YR isn't archived on a generic, so resolve the contract that
        # was front on that date and read its (static) month/year off it.
        resolved = blp.fut_ticker(f"{u_code}1 Comdty", dt=as_of, freq="M")
        if not resolved:
            raise ValueError(f"Could not resolve the {u_code}1 front contract "
                             f"as of {as_of:%Y-%m-%d}")
        contract = full_contract_ticker(resolved, as_of)
        label = blp.bdp(tickers=[contract],
                        flds=["fut_month_yr"])["fut_month_yr"].values[0]
    month, year = label.split(" ")
    return month.capitalize(), year, contract


def fetch_option_chain(u_code: str, as_of=None, n_tenors: int = 6):
    """Option tickers for the first `n_tenors` contracts, as of `as_of`.

    Live: OPT_CHAIN straight off the generics.

    Historical: the generics have rolled since, and OPT_CHAIN is a live-only
    field, so we resolve the contracts that *were* first-to-sixth on that date
    and ask for each one's chain by name. Contracts that have since expired
    return nothing, and those are rebuilt off the strike ladder of the chains we
    did get — strikes that never listed come back empty from bdh and drop out.
    """
    generics = [f"{u_code}{i} Comdty" for i in range(1, n_tenors + 1)]

    if not is_historical(as_of):
        chain = blp.bds(tickers=generics, flds=["OPT_CHAIN"])
        return chain.iloc[:, 0].dropna().tolist()

    contracts = []
    for gen in generics:
        contract = blp.fut_ticker(gen, dt=as_of, freq="M")
        if not contract:
            print(f"[asof] could not resolve {gen} as of {as_of:%Y-%m-%d} - skipped")
            continue
        contracts.append(full_contract_ticker(contract, as_of))

    roots = [option_root(c) for c in contracts]
    chains, expired = {}, []
    for contract, root in zip(contracts, roots):
        got = blp.bds(tickers=[contract], flds=["OPT_CHAIN"])
        tickers = [] if got.empty else got.iloc[:, 0].dropna().tolist()
        if tickers:
            chains[root] = tickers
        else:
            expired.append(root)

    if expired:
        # Ladder source: the still-live contracts of the same product, falling
        # back to today's generics when every as-of contract has expired.
        source = [t for root in chains for t in chains[root]]
        if not source:
            source = blp.bds(tickers=generics, flds=["OPT_CHAIN"]).iloc[:, 0].dropna().tolist()
        ladder = chain_ladder(source)
        if not ladder:
            raise ValueError("No live option chain to rebuild the expired "
                             f"contracts {expired} from")
        print(f"[asof] {', '.join(expired)} have expired - chains rebuilt from a "
              f"{len(ladder)}-strike live ladder")
        for root in expired:
            chains[root] = build_chain_tickers(root, ladder)

    return [tkr for root in roots for tkr in chains[root]]


def fetch_option_oi(option_tickers, as_of=None, chunk: int = 250):
    """{option ticker: open interest} on `as_of` — bdp live, bdh for a past date.

    OPEN_INT is archived, so the past snapshot is real data rather than today's
    number stamped with an old date. The last observation on or before `as_of`
    wins, which rides over holidays and non-printing contracts.
    """
    if not is_historical(as_of):
        data = blp.bdp(tickers=option_tickers, flds="OPEN_INT").dropna(how="all")
        return {} if data.empty else data.dropna().to_dict()["open_int"]

    start = as_of - pd.Timedelta(days=ASOF_LOOKBACK_DAYS)
    oi = {}
    # Chunked: a full chain is a few thousand tickers and one bdh request that
    # wide is slow enough to time out.
    for i in range(0, len(option_tickers), chunk):
        hist = blp.bdh(tickers=option_tickers[i:i + chunk], flds="OPEN_INT",
                       start_date=start, end_date=as_of)
        if hist.empty:
            continue
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(1)  # field level -> ticker columns
        oi.update(hist.ffill().iloc[-1].dropna().to_dict())
    return oi


def fetch_option_greeks(option_tickers, as_of=None):
    """({ticker: gamma}, {ticker: delta}) on `as_of`.

    Empty for a past date: Bloomberg doesn't archive per-contract greeks, so
    there is nothing to fetch and the caller drops the gamma-weighted and
    delta-bucketed views rather than mixing today's greeks into an old snapshot.
    """
    if is_historical(as_of):
        return {}, {}
    df = blp.bdp(tickers=option_tickers, flds=[GAMMA_FIELD, DELTA_FIELD]).to_dict()
    return df.get(GAMMA_FIELD, {}), df.get(DELTA_FIELD, {})


def fetch_spots(underlyings, as_of=None):
    """Last price per underlying on `as_of`, in the order given."""
    if not is_historical(as_of):
        return blp.bdp(tickers=underlyings, flds=["px_last"])["px_last"].dropna().values

    hist = blp.bdh(tickers=underlyings, flds=["px_last"],
                   start_date=as_of - pd.Timedelta(days=ASOF_LOOKBACK_DAYS),
                   end_date=as_of)
    if hist.empty:
        return np.array([])
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.droplevel(1)
    return hist.ffill().iloc[-1].reindex(underlyings).dropna().values


def fetch_twoweek_oi(option_tickers, lookback_days=14, as_of=None):
    """Return (oi_now, oi_prev) OPEN_INT dicts `lookback_days` apart, the later
    one on `as_of` (or today when None).

    OPEN_INT is archived historically (unlike option greeks), so this gives the
    genuine change in open interest per option ticker.
    """
    end = as_of if as_of is not None else pd.Timestamp(datetime.today()).normalize()
    oi_now = fetch_option_oi(option_tickers, as_of=as_of)
    oi_prev = fetch_option_oi(option_tickers,
                              as_of=end - pd.Timedelta(days=lookback_days))
    return oi_now, oi_prev


def oi_change_dict(oi_today, oi_prev):
    """Per-ticker OI change (today - prev). Tickers present on only one day
    treat the missing side as 0 (a newly listed or expiring strike)."""
    tickers = set(oi_today) | set(oi_prev)
    return {t: float(oi_today.get(t, 0.0)) - float(oi_prev.get(t, 0.0)) for t in tickers}


def generate_options_oi_plots(
        underlying_str="Brent",
        compare_prev_2week=False,
        show_by_delta=False,
        _date=None
):
    """Render the options open-interest chart set for `underlying_str`.

    _date : snapshot date. None (or today) pulls the latest values off bds/bdp.
            A past date pulls open interest as of that date with bdh, and names
            the PNGs with a _YYYYMMDD suffix so a back-dated run never overwrites
            the live ones. Bloomberg archives no per-contract greeks, so the
            gamma-weighted and delta-bucketed charts are skipped on a past date.
    """
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
            "Brent": [20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 150, 200],
            "WTI": [20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 150, 200],
            "TTF": [5, 10, 15, 20, 25, 30,40, 45, 50, 55, 60, 65, 70, 80, 90, 100],
            "HH": [0.1, 0.5, 1.0, 1.5, 2.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0]
        }[underlying_str]

        sym_levels = {
            "Brent": [], #[1.0, 2.0, 3.0, 5.0],
            "WTI": [], # [1.0, 2.0, 3.0, 5.0],
            "TTF": [0.5, 1.0, 2.0, 3.0, 5.0],
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
    as_of = as_of_timestamp(_date)
    historical = is_historical(as_of)
    # Back-dated PNGs get their own filenames, so re-running history can't
    # clobber the images the daily report picks up.
    suffix = f"_{as_of:%Y%m%d}" if historical else ""
    stamp = f" ({as_of:%Y-%m-%d})" if historical else ""
    when = f"{as_of:%Y-%m-%d}" if as_of is not None else "today"
    if historical:
        print(f"[asof] snapshot as of {as_of:%Y-%m-%d}: open interest from bdh; "
              f"greeks are not archived, so gamma/delta views are skipped")

    month, year, front_contract = fetch_front_contract(u_code, as_of)
    year_code = year[-1] if underlying_str != "HH" else year
    bbg_month_code = bbg_dict[month]

    # Front-month options aggregate. A back-dated run takes the contract ticker
    # resolved off `as_of`: the short-year form built below ('COQ5 Comdty') stops
    # returning history once the contract is a year old, where the two-digit
    # form ('COQ25 Comdty') still has it.
    TICKERS = [front_contract or f"{u_code}{bbg_month_code}{year_code} Comdty"]
    FIELDS = ["OPEN_INT_TOTAL_PUT", "OPEN_INT_TOTAL_CALL", "OPEN_INT",
              "AGGREGATE_PUT_OPEN_INT", "AGGREGATE_CALL_OPEN_INT", "FUT_AGGTE_OPEN_INT"]
    dates = pd.date_range(end=as_of if historical else datetime.today(), periods=300)
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
        underlying=f"{underlying_str}{stamp}",
        active_contract=f"{month}-{year}",
        savepath=f"C:\\Marti\\{underlying_str}_P_C_history{suffix}.png"
    )

    underlyings = [f"{u_code}{i} Comdty" for i in [1, 2, 3, 4, 5, 6]]

    option_tickers = fetch_option_chain(u_code, as_of)
    spots = fetch_spots(underlyings, as_of)
    if not len(spots):
        raise ValueError(f"No {underlying_str} underlying prices as of "
                         f"{when} - is it a trading day?")

    # Open interest, plus the per-contract greeks when they exist, so we can
    # weight OI by gamma and bucket it by delta. (Adjust GAMMA_FIELD /
    # DELTA_FIELD if your terminal names the greeks differently.)
    oi_dict = fetch_option_oi(option_tickers, as_of)
    gamma_dict, delta_dict = fetch_option_greeks(option_tickers, as_of)
    if not oi_dict:
        raise ValueError(f"No {underlying_str} option open interest as of "
                         f"{when} - is it a trading day?")

    fig = plot_oi_by_tenor(
        oi_dict,
        ticker=f"{underlying_str}{stamp}",
        bucket_width=bucket_width,
        savepath=f"C:\\Marti\\{underlying_str}_OI_by_tenor{suffix}.png",
        spots=spots,
        max_strike=max_strike, min_strike=min_strike,
    )
    if gamma_dict:
        fig_gamma = plot_oi_by_tenor(
            oi_dict,
            ticker=f"{underlying_str}{stamp}",
            bucket_width=bucket_width,
            savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_by_tenor{suffix}.png",
            spots=spots,
            max_strike=max_strike, min_strike=min_strike,
            weight="gamma", gamma_dict=gamma_dict,
        )

    spot = spots[0]
    spot_int = int(spot)
    anchors = [spot_int + i for i in sym_levels] + [spot_int - i for i in sym_levels] + [spot_int] + fixed_strikes_anchors
    anchors = sorted(anchors)

    for mode in ["total"]:  # ["net", "total"]:
        # Raw open-interest grid.
        _ = plot_grid(
            oi_dict,
            savepath=f"C:\\Marti\\{underlying_str}_OI_options_{mode}{suffix}.png",
            underlying=f"{underlying_str}{stamp}",
            mode=mode,
            spot=spot,
            anchors=anchors,
        )
        # Gamma-weighted grid. Live-only: the greeks aren't archived.
        if gamma_dict:
            _ = plot_grid(
                oi_dict,
                savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_options_{mode}{suffix}.png",
                underlying=f"{underlying_str}{stamp}",
                mode=mode,
                spot=spot,
                anchors=anchors,
                weight="gamma",
                gamma_dict=gamma_dict,
            )
        # Same aggregation in moneyness space: 5% delta clusters instead of
        # absolute strikes, so the wings stay comparable as the forward moves.

        # _ = plot_delta_grid(
        #     oi_dict, delta_dict,
        #     savepath=f"C:\\Marti\\{underlying_str}_OI_options_by_delta_{mode}.png",
        #     underlying=underlying_str,
        #     mode=mode,
        # )

        if show_by_delta and delta_dict and gamma_dict:
            _ = plot_delta_grid(
                oi_dict, delta_dict,
                savepath=f"C:\\Marti\\{underlying_str}_gamma_OI_options_by_delta_{mode}{suffix}.png",
                underlying=f"{underlying_str}{stamp}",
                mode=mode,
                weight="gamma",
                gamma_dict=gamma_dict,
            )

    if compare_prev_2week:
        # Day-over-day: real OI change from bdh, gamma held at today's values so
        # the picture isolates the positioning shift (Γ·ΔOI). Strikes that only
        # existed yesterday have no today-gamma and drop out (contribute 0).
        oi_today_h, oi_prev_h = fetch_twoweek_oi(option_tickers, as_of=as_of)
        oi_change = oi_change_dict(oi_today_h, oi_prev_h)

        for mode in ["total"]:  # ["net", "total"]:
            _ = plot_grid(
                oi_change,
                savepath=f"C:\\Marti\\{underlying_str}_OI_change_{mode}{suffix}.png",
                underlying=f"{underlying_str} Δ15 days{stamp}",
                mode=mode,
                spot=spot,
                anchors=anchors,
                # weight="gamma",
                gamma_dict=gamma_dict,
                center_zero=True,
            )
        _ = plot_oi_by_tenor(
            oi_change,
            ticker=f"{underlying_str} Δ15d{stamp}",
            bucket_width=bucket_width,
            savepath=f"C:\\Marti\\{underlying_str}_OI_change_by_tenor{suffix}.png",
            spots=spots,
            max_strike=max_strike, min_strike=min_strike,
            # weight="gamma",
            gamma_dict=gamma_dict,
        )


if __name__ == "__main__":
    a = 1
    generate_options_oi_plots(
        underlying_str="Brent",
        compare_prev_2week=True,
        _date=None,  # or e.g. "2026-06-15" for a back-dated snapshot
    )