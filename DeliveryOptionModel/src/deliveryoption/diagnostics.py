"""Diagnostics for a calibration folder: how the data behaved, tenor by tenor.

    python diagnose.py                      # latest calibration
    python diagnose.py --calib 2026-09-24   # a named one

Writes <calibration>/diagnostics/:

    index.md                    what each plot shows, plus findings computed from the data
    vol_term_structure.png      ATM vol by delivery month and hub, by source (live mid /
                                settlement / filled), against the realised vol history
                                gives for the same maturity
    vol_profile.png             realised vol by days to expiry (the Samuelson shape)
    corr_term_structure.png     M/M+1 and cross-hub correlation along the strip,
                                weekly (used in pricing) and daily (diagnostic)
    corr_<series>.png           one per series (TTF_M/TTF_M1, TTF_M/HH_M, ...): for every
                                tenor pair, the pooled correlation broken down by calendar
                                year, by each year's share of the variance (what drives a
                                pooled number), by days to expiry and by which month pair
                                supplied the data
    data_quality.png            observations, window length and stale prints per pair
    tables/*.csv                the numbers behind every plot

The breakdowns re-run the calibration's own sampling (calibration.window_returns)
on the price history the calibration job saves in <calibration>/history/, so
the slices add up to the numbers the pricer uses. A calibration written before
history was saved is re-fetched from Bloomberg once, on first run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors                                             # noqa: E402
import matplotlib.ticker                                             # noqa: E402
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402

from DeliveryOptionModel.src.deliveryoption import calibration, snapshot   # noqa: E402
from DeliveryOptionModel.src.deliveryoption.market_data import hub_contract  # noqa: E402
from DeliveryOptionModel.src.deliveryoption.vols import profile_vol   # noqa: E402

HISTORY_DIR = "history"
OUT_DIR = "diagnostics"

# Buckets for the slices; a cell with fewer weekly returns than this is left
# blank rather than shown as a correlation of noise.
TAU_EDGES = [0, 30, 60, 90, 180, 365, 730, 1095, 1460, 2200]
MIN_CELL_OBS = 8

# --- palette: the dataviz reference instance ---------------------------------
# Hubs take categorical slots in registry order. Only the first three slots
# are validated for all pairs (worst CVD dE 9.2 light), which covers TTF / HH /
# Brent; a fourth hub would need small multiples rather than a fourth hue.
HUB_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
MID_GRAY = "#f0efec"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
DIV = LinearSegmentedColormap.from_list("div_red_blue", ["#e34948", MID_GRAY, "#2a78d6"])
for _cm in (SEQ, DIV):
    _cm.set_bad(SURFACE)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.fontsize": 8,
    "savefig.dpi": 150, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def save_history(folder: Path, px: pd.DataFrame, expiry: pd.Series):
    """Keep the prices the calibration used, so diagnostics can be rebuilt offline.

    Raw Bloomberg data: .gitignore keeps it out of the repository.
    """
    d = folder / HISTORY_DIR
    d.mkdir(parents=True, exist_ok=True)
    px.to_parquet(d / "prices.parquet")
    expiry.rename("expiry").rename_axis("ticker").to_csv(d / "expiry.csv")


def load_history(folder: Path) -> tuple[pd.DataFrame, pd.Series]:
    d = folder / HISTORY_DIR
    px = pd.read_parquet(d / "prices.parquet")
    expiry = pd.read_csv(d / "expiry.csv", index_col=0, parse_dates=["expiry"])["expiry"]
    return px, expiry


def fetch_missing_history(folder: Path, snap: snapshot.Snapshot, pairs: pd.DataFrame):
    """Rebuild the calibration's history pull from its own settings (Bloomberg)."""
    cal, hub_specs = snap.meta["calibration"], snap.meta["hubs"]
    hist_months, reach = set(), cal["vol_profile_buckets_days"][-1]
    for p in pairs.itertuples():
        anchors = calibration.anchor_months(p.m0, snap.as_of, cal["lookback_months"],
                                            cal["anchor_filter"] == "same_pair")
        hist_months |= {m for k in anchors for m in (k, k + p.gap)}
        reach = max(reach, p.window_hi)
    start = min(hist_months).start_time - pd.Timedelta(days=reach + 31)
    print(f"No saved history in {folder.name}: fetching {len(hist_months) * len(hub_specs)} "
          f"contracts from Bloomberg (once)")
    px, expiry = calibration.fetch_history(hub_specs, list(hub_specs), hist_months, start, snap.as_of)
    save_history(folder, px, expiry)
    return px, expiry


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def calibrated_pairs(folder: Path) -> pd.DataFrame:
    s = pd.read_csv(folder / "correlation_summary.csv")
    if "error" in s:
        s = s[s["error"].isna()]
    m = s["pair"].str.split("_", expand=True)
    s["m0"] = m[0].map(lambda x: pd.Period(x, "M"))
    s["m1"] = m[1].map(lambda x: pd.Period(x, "M"))
    s["gap"] = [(b - a).n for a, b in zip(s["m0"], s["m1"])]
    return s.reset_index(drop=True)


def series_names(hubs: list[str]) -> tuple[list[str], list[str]]:
    within = [f"{h}_M/{h}_M1" for h in hubs]
    cross = [f"{a}_M/{b}_M" for i, a in enumerate(hubs) for b in hubs[i + 1:]]
    return within, cross


def correlation_slices(px, expiry, snap, pairs: pd.DataFrame) -> pd.DataFrame:
    """Every calibrated pair's pooled sample, sliced by year, maturity and month pair.

    Long table: pair, m0, series, dim, bucket, n, corr, var_share. `var_share`
    is the bucket's share of the sum of squared returns of the two legs: in a
    pooled correlation a bucket weighs in proportion to it.
    """
    cal, hub_specs = snap.meta["calibration"], snap.meta["hubs"]
    hubs = list(hub_specs)
    within, cross = series_names(hubs)
    rows = []
    for p in pairs.itertuples():
        anchors = calibration.anchor_months(p.m0, snap.as_of, cal["lookback_months"],
                                            cal["anchor_filter"] == "same_pair")
        try:
            r = calibration.window_returns(px, expiry, hub_specs, hubs, anchors, p.gap,
                                           (p.window_lo, p.window_hi), cal["return_days"], snap.as_of)
        except ValueError:
            continue
        anchor = r.index.get_level_values("anchor")
        dates = r.index.get_level_values("date")
        fix = {k: min(expiry[hub_contract(hub_specs[h], pd.Period(k, "M"))] for h in hubs)
               for k in anchor.unique()}
        tau = (pd.Series(anchor.map(fix), index=r.index) - dates).dt.days.to_numpy()
        keys = {
            "year": dates.year.astype(str),
            "days_to_expiry": pd.cut(tau, TAU_EDGES, right=True).astype(str),
            "month_pair": [f"{pd.Period(k, 'M').strftime('%b')}/{(pd.Period(k, 'M') + p.gap).strftime('%b')}"
                           for k in anchor],
            "pooled": np.full(len(r), "all"),
        }
        for s in within + cross:
            a, b = s.split("/")
            x, y = r[a].to_numpy(), r[b].to_numpy()
            ss = x ** 2 + y ** 2
            total = ss.sum()
            for dim, key in keys.items():
                df = pd.DataFrame({"x": x, "y": y, "ss": ss, "key": key})
                for bucket, g in df.groupby("key", sort=False):
                    n = len(g)
                    rows.append({"pair": p.pair, "m0": str(p.m0), "series": s, "dim": dim,
                                 "bucket": bucket, "n": n,
                                 "corr": g.x.corr(g.y) if n >= MIN_CELL_OBS else np.nan,
                                 "var_share": g.ss.sum() / total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def vol_table(snap: snapshot.Snapshot) -> pd.DataFrame:
    c = snap.contracts.reset_index()
    src = c["vol_source"].fillna("").astype(str)
    c["source"] = np.select(
        [src.str.contains("IVOL_MID"), src.str.contains("IVOL_LAST"), c["atm_vol"].isna() & c["vol"].notna()],
        ["live mid", "settlement", "filled"], "missing")
    expiry = pd.to_datetime(c["option_expiry"]).fillna(pd.to_datetime(c["fixing"]))
    c["days_to_expiry"] = (expiry - snap.as_of).dt.days
    prof = snap.vol_profile
    c["realised_same_maturity"] = [
        profile_vol(prof[prof["hub"] == h], d) if (prof["hub"] == h).any() and d > 0 else np.nan
        for h, d in zip(c["hub"], c["days_to_expiry"])]
    return c[["hub", "month", "future", "fwd", "days_to_expiry", "atm_vol", "vol", "source",
              "realised_same_maturity"]]


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _footnote(fig, text: str, width: float):
    """Footnote wrapped to the figure width (inches) so it is never cut off."""
    import textwrap
    fig.text(0.01, 0.005, textwrap.fill(text, width=int(width * 17)), color=INK_2, fontsize=8,
             va="bottom")


def _colors(hubs):
    if len(hubs) > len(HUB_SLOTS):
        raise ValueError(f"{len(hubs)} hubs: only {len(HUB_SLOTS)} validated hub colours")
    return dict(zip(hubs, HUB_SLOTS))


def plot_vol_term(vt: pd.DataFrame, hubs, path: Path):
    colors = _colors(hubs)
    fig, axes = plt.subplots(len(hubs), 1, figsize=(10, 2.6 * len(hubs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, hub in zip(axes, hubs):
        d = vt[vt["hub"] == hub].copy()
        d["t"] = d["month"].map(lambda m: pd.Period(m, "M").to_timestamp())
        ax.plot(d["t"], d["realised_same_maturity"], color=MUTED, linewidth=2,
                label="realised history, same maturity")
        styles = {"live mid": dict(marker="o", facecolors=colors[hub], edgecolors=SURFACE),
                  "settlement": dict(marker="o", facecolors=SURFACE, edgecolors=colors[hub]),
                  "filled": dict(marker="X", facecolors=colors[hub], edgecolors=SURFACE)}
        for source, st in styles.items():
            s = d[d["source"] == source]
            if len(s):
                ax.scatter(s["t"], s["vol"], s=36, linewidths=1.5, label=f"{source} ({len(s)})",
                           zorder=3, **st)
        ax.set_title(f"{hub}: ATM vol by delivery month")
        _tidy(ax)
        _pct(ax)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", ncol=4)
    _footnote(fig, "Marker = where the vol came from: filled circle live mid (IVOL_MID), "
              "hollow circle exchange settlement (IVOL_LAST), cross filled from the realised curve. "
              "Grey line = the realised vol history gives a contract of the same maturity.", width=10)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def plot_vol_profile(profile: pd.DataFrame, hubs, path: Path):
    colors = _colors(hubs)
    fig, ax = plt.subplots(figsize=(10, 4))
    cap = TAU_EDGES[-1]
    for hub in hubs:
        p = profile[profile["hub"] == hub].sort_values("tau_lo_days")
        if p.empty:
            continue
        x = list(p["tau_lo_days"]) + [min(float(p["tau_hi_days"].iloc[-1]), cap)]
        y = list(p["realised_vol"]) + [p["realised_vol"].iloc[-1]]
        ax.step(x, y, where="post", color=colors[hub], linewidth=2, label=hub)
        ax.annotate(hub, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8)
    ax.set_title("Realised vol by days to expiry (weekly returns, pooled over contracts)")
    ax.set_xlabel("days to the contract's expiry")
    ax.set_xlim(0, cap + 150)
    _tidy(ax)
    _pct(ax)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_corr_term(pairs: pd.DataFrame, hubs, path: Path):
    """Left: M/M+1 per hub (hub colours). Right: one small panel per cross-hub
    pair in neutral ink - a cross pair is not one hub, so it gets no hub colour."""
    colors = _colors(hubs)
    within, cross = series_names(hubs)
    cross = [c for c in cross if c in pairs]
    t = pairs["m0"].map(lambda m: m.to_timestamp())
    style = [Line2D([], [], color=INK_2, linewidth=2), Line2D([], [], color=INK_2, linewidth=1, alpha=0.45)]
    style_labels = ["weekly (used in pricing)", "daily (diagnostic)"]

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(max(len(cross), 1), 2, width_ratios=[1.35, 1], hspace=0.55, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    for h, s_ in zip(hubs, within):
        if s_ not in pairs:
            continue
        ax.plot(t, pairs[s_], color=colors[h], linewidth=2, label=h)
        if f"{s_} daily" in pairs:
            ax.plot(t, pairs[f"{s_} daily"], color=colors[h], linewidth=1, alpha=0.45)
    ax.set_title("M / M+1 correlation, same hub", pad=26)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + style, labels + style_labels, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=5, borderaxespad=0.2)
    ax.set_xlabel("first delivery month of the pair")
    _tidy(ax)

    for i, s_ in enumerate(cross):
        a = fig.add_subplot(gs[i, 1], sharex=ax)
        a.plot(t, pairs[s_], color=INK_2, linewidth=2)
        if f"{s_} daily" in pairs:
            a.plot(t, pairs[f"{s_} daily"], color=INK_2, linewidth=1, alpha=0.45)
        hub_a, hub_b = (x.split("_")[0] for x in s_.split("/"))
        a.set_title(f"cross hub, M / M: {hub_a} / {hub_b}", fontsize=9)
        if i == 0:
            a.legend(style, style_labels, loc="lower left", bbox_to_anchor=(0, 1.18), ncol=2,
                     borderaxespad=0.2)
        if i < len(cross) - 1:
            plt.setp(a.get_xticklabels(), visible=False)
        _tidy(a)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.12)
    fig.savefig(path)
    plt.close(fig)


BUCKET_ORDER = {
    "days_to_expiry": [str(pd.Interval(a, b, closed="right")) for a, b in zip(TAU_EDGES, TAU_EDGES[1:])],
}


def plot_corr_slices(sl: pd.DataFrame, series: str, path: Path, within: bool):
    """Four heatmaps, one row per tenor pair: what the pooled number is made of."""
    d = sl[sl["series"] == series]
    if d.empty:
        return
    pairs = sorted(d["m0"].unique())
    pooled = d[d["dim"] == "pooled"].set_index("m0")["corr"].reindex(pairs)

    def grid(dim, value):
        g = d[d["dim"] == dim].pivot_table(index="m0", columns="bucket", values=value, aggfunc="first")
        cols = BUCKET_ORDER.get(dim)
        if dim == "month_pair":
            order = [m.strftime("%b") for m in pd.period_range("2000-01", "2000-12", freq="M")]
            cols = sorted(g.columns, key=lambda c: order.index(c.split("/")[0]))
        return g.reindex(index=pairs, columns=cols if cols else sorted(g.columns))

    panels = [("year", "corr", "by calendar year of the return"),
              ("year", "var_share", "share of the variance, by year"),
              ("days_to_expiry", "corr", "by days to expiry of M"),
              ("month_pair", "corr", "by month pair supplying the data")]
    grids = [grid(dim, v) for dim, v, _ in panels]
    corr_vals = np.concatenate([g.to_numpy().ravel() for g, (_, v, _) in zip(grids, panels) if v == "corr"]
                               + [pooled.to_numpy()])
    corr_vals = corr_vals[np.isfinite(corr_vals)]
    if within:
        lo = min(0.9, float(np.percentile(corr_vals, 2))) if len(corr_vals) else 0.9
        corr_norm = matplotlib.colors.Normalize(vmin=round(lo, 2), vmax=1.0)
        corr_cmap = SEQ
    else:
        span = float(np.nanmax(np.abs(corr_vals))) if len(corr_vals) else 1.0
        corr_norm = TwoSlopeNorm(vcenter=0.0, vmin=-span, vmax=span)
        corr_cmap = DIV

    # Explicit layout: every heatmap spans the same vertical extent so a row is
    # the same tenor pair in every panel; colourbars sit in their own strip.
    n_rows = len(pairs)
    W, H = 16.0, max(5.0, 0.17 * n_rows + 3.0)
    top, bottom = 1 - 0.95 / H, 1.75 / H
    cb_y, cb_h = 0.62 / H, 0.11 / H
    left, right, gap = 0.06, 0.99, 0.014
    widths = np.array([0.9] + [max(g.shape[1], 3) for g in grids], dtype=float)
    widths = widths / widths.sum() * (right - left - gap * (len(widths) - 1))
    xs = left + np.concatenate([[0], np.cumsum(widths[:-1] + gap)])
    fig = plt.figure(figsize=(W, H))
    extend = "min" if within else "neither"

    def heat(ax, values, cmap, norm):
        return ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax = fig.add_axes([xs[0], bottom, widths[0], top - bottom])
    heat(ax, pooled.to_numpy()[:, None], corr_cmap, corr_norm)
    ax.set_title("pooled\n(used)")
    ax.set_xticks([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([pd.Period(m, "M").strftime("%b-%y") for m in pairs], fontsize=7)
    ax.set_ylabel("tenor pair: first month")
    axes = [ax]

    for x, w, g, (dim, value, title) in zip(xs[1:], widths[1:], grids, panels):
        ax = fig.add_axes([x, bottom, w, top - bottom])
        if value == "var_share":
            vmax = max(0.05, float(np.nanmax(g.to_numpy())))
            im = heat(ax, g.to_numpy(), SEQ, matplotlib.colors.Normalize(0, vmax))
        else:
            im = heat(ax, g.to_numpy(), corr_cmap, corr_norm)
        ax.set_title(title)
        labels = [c.replace("(", "").replace("]", "").replace(", ", "-") for c in g.columns]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45 if dim == "year" else 90, fontsize=7,
                           ha="right" if dim == "year" else "center")
        ax.set_yticks([])
        cax = fig.add_axes([x, cb_y, w, cb_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                          extend="neither" if value == "var_share" else extend)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_2, length=0)
        if value == "var_share":
            cb.ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axes.append(ax)
    for ax in axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    kind = "same hub, M vs M+1" if within else "cross hub, M vs M"
    fig.suptitle(f"{series}: weekly-return correlation ({kind}) for each tenor pair, "
                 "and what the pooled number is made of", x=0.01, y=1 - 0.25 / H, ha="left",
                 fontsize=11, fontweight="semibold", color=INK)
    note = (f"Rows are the calibrated tenor pairs (a gap in the months = not calibrated). Blank cell = "
            f"fewer than {MIN_CELL_OBS} weekly returns. The pooled column uses the correlation colour scale. "
            + (f"Correlations below {corr_norm.vmin:.2f} show at the lightest step (arrow on the scale). "
               if within else "Red negative, grey zero, blue positive. ")
            + "A pooled correlation weighs each year by its share of the variance (2nd panel).")
    _footnote(fig, note, width=W)
    fig.savefig(path)
    plt.close(fig)


def plot_data_quality(pairs: pd.DataFrame, path: Path):
    t = pairs["m0"].map(lambda m: m.to_timestamp())
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    series = [("n_obs", "weekly returns in the pooled sample", None),
              ("window_hi", "window length, days before fixing", None),
              ("max_stale_share", "worst leg's share of zero weekly returns (stale prints)", "pct")]
    for ax, (col, title, fmt) in zip(axes, series):
        if col not in pairs:
            continue
        ax.plot(t, pairs[col], color=HUB_SLOTS[0], linewidth=2, marker="o", markersize=3.5,
                markeredgecolor=SURFACE)
        ax.set_title(title)
        _tidy(ax)
        if fmt == "pct":
            _pct(ax, decimals=1)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel("first delivery month of the pair")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# findings + index
# ---------------------------------------------------------------------------

def findings(vt, sl, pairs, hubs) -> list[str]:
    """A few facts worth a look, computed - not a substitute for the plots."""
    out = []
    for hub in hubs:
        v = vt[vt["hub"] == hub]
        counts = v["source"].value_counts()
        filled = v[v["source"] == "filled"]
        text = (f"**{hub} vols:** " + ", ".join(f"{counts.get(s, 0)} {s}" for s in
                                                ("live mid", "settlement", "filled", "missing")))
        if len(filled):
            text += f"; filled from {filled['month'].min()} onwards"
        out.append(text + ".")
    within, cross = series_names(hubs)
    for s in within:
        if s not in pairs:
            continue
        lo = pairs.loc[pairs[s].idxmin()]
        out.append(f"**{s}:** {pairs[s].min():.3f} to {pairs[s].max():.3f} along the strip "
                   f"(lowest {pd.Period(lo['m0'], 'M').strftime('%b-%y')}).")
        d = sl[(sl["series"] == s) & (sl["dim"] == "year")]
        if not d.empty:
            top = d.groupby("bucket")["var_share"].mean().sort_values(ascending=False)
            corr_by_year = d.groupby("bucket")["corr"].mean()
            out.append(f"  - variance is dominated by {top.index[0]} "
                       f"({top.iloc[0]:.0%} of it on average), when this correlation averaged "
                       f"{corr_by_year[top.index[0]]:.3f}.")
        m = sl[(sl["series"] == s) & (sl["dim"] == "month_pair")]
        if not m.empty:
            by = m.groupby("bucket")["corr"].mean().dropna().sort_values()
            if len(by):
                out.append(f"  - weakest month pairs: " + ", ".join(
                    f"{k} {v:.3f}" for k, v in by.head(3).items()) + ".")
    if "max_stale_share" in pairs:
        worst = pairs.loc[pairs["max_stale_share"].idxmax()]
        out.append(f"**Stale prints:** at most {worst['max_stale_share']:.1%} zero weekly returns "
                   f"({worst['pair']}).")
    return out


INDEX = """# Calibration diagnostics - {name}

As of {as_of}. Built from this folder's files and the price history the calibration
used ({history}). Numbers behind every plot are in `tables/`.

## Findings (computed)

{findings}

## Plots

| file | what it shows | what to look for |
|---|---|---|
| `vol_term_structure.png` | ATM vol per delivery month and hub, marked by source; grey = realised vol history gives the same maturity | where market vols stop and fills start; implied far above realised = a rich market |
| `vol_profile.png` | realised vol by days to expiry | the Samuelson shape the horizon adjustment uses; a flat curve means little is stripped from M+1 |
| `corr_term_structure.png` | M/M+1 and cross-hub correlation along the strip | weekly vs daily gap (close asynchrony); level drift with window length |
| `corr_<series>.png` | per tenor pair: pooled correlation and its breakdown by year, variance share, days to expiry, month pair | which years carry the variance; season-change month pairs (Mar/Apr, Sep/Oct) pulling a pair down |
| `data_quality.png` | sample size, window length and stale prints per pair | thin samples, stale far-dated prints |

Colours follow the hub: {colors}. Heatmaps: one blue ramp for magnitudes, red-grey-blue
for correlations that can be negative.
"""


def run(folder: Path, fetch: bool = True) -> Path:
    folder = Path(folder)
    snap = snapshot.read(folder)
    hubs = list(snap.meta["hubs"])
    pairs = calibrated_pairs(folder)
    if (folder / HISTORY_DIR / "prices.parquet").exists():
        px, expiry = load_history(folder)
        history = f"`{HISTORY_DIR}/`"
    elif fetch:
        px, expiry = fetch_missing_history(folder, snap, pairs)
        history = f"`{HISTORY_DIR}/`, re-fetched from Bloomberg"
    else:
        raise FileNotFoundError(f"no {HISTORY_DIR}/ in {folder} and fetching is off")

    out = folder / OUT_DIR
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    vt = vol_table(snap)
    sl = correlation_slices(px, expiry, snap, pairs)
    vt.to_csv(tables / "vol_term_structure.csv", index=False)
    snap.vol_profile.to_csv(tables / "vol_profile.csv", index=False)
    pairs.drop(columns=["m0", "m1"]).to_csv(tables / "correlation_term_structure.csv", index=False)
    sl.to_csv(tables / "correlation_slices.csv", index=False)

    plot_vol_term(vt, hubs, out / "vol_term_structure.png")
    plot_vol_profile(snap.vol_profile, hubs, out / "vol_profile.png")
    plot_corr_term(pairs, hubs, out / "corr_term_structure.png")
    within, cross = series_names(hubs)
    for s in within + cross:
        plot_corr_slices(sl, s, out / f"corr_{s.replace('/', '_vs_')}.png", within=s in within)
    plot_data_quality(pairs, out / "data_quality.png")

    colors = ", ".join(f"{h} `{c}`" for h, c in _colors(hubs).items())
    (out / "index.md").write_text(INDEX.format(
        name=folder.name, as_of=snap.meta["as_of"], history=history, colors=colors,
        findings="\n".join(f"- {f}" if not f.startswith("  ") else f
                           for f in findings(vt, sl, pairs, hubs))), encoding="utf-8")
    return out


def main(argv=None):
    import argparse
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default="latest", help="'latest', a folder name, or a path")
    ap.add_argument("--calib-root", default=str(root / "data" / "calibration"))
    ap.add_argument("--no-fetch", action="store_true", help="fail rather than fetch missing history")
    args = ap.parse_args(argv)
    folder = snapshot.resolve(Path(args.calib_root), args.calib)
    out = run(folder, fetch=not args.no_fetch)
    print(f"Diagnostics written to {out}")
    print(json.dumps(sorted(p.name for p in out.glob("*.png")), indent=1))


if __name__ == "__main__":
    main()
