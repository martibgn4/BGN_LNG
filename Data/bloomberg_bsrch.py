
from xbbg import blp
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from Data.running_lng_on_water import run_lng_on_water_processing

"""
LNG Shipping Daily Recap
------------------------
Aggregates Bloomberg LNG vessel data into:
  1. Imports by destination region (with breakdown by origin region)
  2. On-Transit cargoes (sailed but not yet arrived)

Usage:
    python lng_recap.py <input_csv>
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse dates
    for col in ["Arrival", "Departure", "Origin_Departuredate", "Destination_Arrivaldate"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Quantities are negative for imports in the source; use absolute values
    for col in ["Estimated_Qty", "Qtymtonnes", "Qtymcm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()

    # Drop bunkering / small ship-to-ship noise — keep only real LNG cargoes
    df = df[df["Vessel_Type"].str.contains("LNG Tanker", case=False, na=False)].copy()

    return df


# ---------------------------------------------------------------------------
# Aggregation 1: Imports by destination region
# ---------------------------------------------------------------------------
def imports_by_region(df, start_date=None, end_date=None) -> pd.DataFrame:
    """One row per destination region: cargo count, total mtonnes, total mcm."""
    imp = df[df["Visit_Type"].str.lower() == "import"].copy()
    if start_date is not None and end_date is not None:
        imp = imp[(imp["Arrival"] >= start_date) & (imp["Arrival"] <= end_date)]

    agg = (
        imp.groupby("Port_Region", dropna=False)
        .agg(Cargoes=("Imo", "count"),
             Mtonnes=("Qtymtonnes", "sum"),
             Mcm=("Qtymcm", "sum"))
        .round(2)
        .sort_values("Mtonnes", ascending=True)
    )
    agg.loc["TOTAL"] = agg.sum(numeric_only=True)
    return agg


def imports_origin_destination(df, start_date=None, end_date=None, index="Origin_Berthcountry", aggfunc="sum") -> pd.DataFrame:
    """Pivot of mtonnes flowing from each origin region to each destination region."""
    imp = df[df["Visit_Type"].str.lower() == "import"].copy()

    if start_date is not None and end_date is not None:
        imp = imp[(imp["Arrival"] >= start_date) & (imp["Arrival"] <= end_date)]

    pivot = (
        imp.pivot_table(index=index,
                        columns="Port_Region",
                        values="Qtymtonnes",
                        aggfunc=aggfunc,
                        fill_value=0)
        .round(2)
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot.loc["TOTAL"] = pivot.sum(axis=0)
    return pivot.sort_values("TOTAL", ascending=True)


# ---------------------------------------------------------------------------
# Aggregation 2: On-transit (sailed, not yet arrived)
# ---------------------------------------------------------------------------
def on_transit(df: pd.DataFrame) -> pd.DataFrame:
    """Cargoes flagged On Transit — show where from, when sailed, going where."""
    transit = df[df["Destination_Berthport"].str.lower().str.contains("transit", na=False)].copy()

    cols = ["Vessel_Name", "Imo",
            "Origin_Departuredate", "Origin_Berthport", "Origin_Berthcountry", "Origin_Berthregion",
            "Destination_Arrivaldate", "Destination_Berthport", "Destination_Berthcountry", "Destination_Berthregion",
            "Qtymtonnes", "Qtymcm"]
    cols = [c for c in cols if c in transit.columns]

    out = (transit[cols]
           .sort_values("Origin_Departuredate")
           .reset_index(drop=True))
    return out


def transit_by_origin(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of on-transit cargoes by origin region."""
    transit = df[df["Destination_Berthport"].str.lower().str.contains("transit", na=False)].copy()
    if transit.empty:
        return pd.DataFrame()

    agg = (transit.groupby("Port_Country", dropna=False)
           .agg(Cargoes=("Imo", "count"),
                Mtonnes=("Qtymtonnes", "sum"),
                Mcm=("Qtymcm", "sum"))
           .round(2)
           .sort_values("Mtonnes", ascending=False))
    agg.loc["TOTAL"] = agg.sum(numeric_only=True)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_recap(import_df, export_df, import_rolling_values=[1, 7, 30], od_index="Origin_Berthcountry", aggfunc="sum"):
    from datetime import timedelta
    today = datetime.today().date()
    imports_summary_last_n_days = {}
    od_matrix_last_n_days = {}
    for d in import_rolling_values:

        imports_summary = imports_by_region(
            import_df,
            start_date=today - timedelta(d),
            end_date=today #- timedelta(1)
        )

        od_matrix = imports_origin_destination(
            import_df,
            start_date=today - timedelta(d),
            end_date=today, #- timedelta(1),
            index=od_index,
            aggfunc=aggfunc
        )

        imports_summary_last_n_days[d] = imports_summary
        od_matrix_last_n_days[d] = od_matrix


    transit_more_than_n_days = {}
    for n in [1, 20, 30]:
        export_df_n_days = export_df[export_df["Departure"] <= today - timedelta(n-1)]
        transit_summary = transit_by_origin(export_df_n_days)
        transit_more_than_n_days[n] = transit_summary



    return imports_summary_last_n_days, od_matrix_last_n_days, transit_more_than_n_days

ORIGIN_PALETTE = [
    "#E87722",  # orange  (Qatar-style)
    "#3CB371",  # green   (Oman-style)
    "#9B8CFA",  # purple  (UAE-style)
    "#1F77B4",  # blue
    "#D62728",  # red
    "#8C564B",  # brown
    "#17BECF",  # cyan
    "#BCBD22",  # olive
    "#7F7F7F",  # gray
]

def hex_to_rgba(hex_color: str, alpha: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def matrix_to_flows(flows_matrix: pd.DataFrame, neg_magnitudes=True) -> pd.DataFrame:
    """Convert a wide origin x destination matrix into a long flow table.

    Strips any TOTAL row/column, drops zero/NaN entries, and returns a
    DataFrame with columns ['origin', 'destination', 'value'] sorted by
    descending value.
    """
    m = flows_matrix.copy()

    # Drop common total rows/columns that often hang on the matrix
    drop_labels = {"TOTAL", "Total", "total", "All", "Sum", "TOTAL.", "Total.", "total.", "All.", "Sum.", "Bunkering."}
    m = m.drop(index=[i for i in m.index if i in drop_labels], errors="ignore")
    m = m.drop(columns=[c for c in m.columns if c in drop_labels], errors="ignore")

    long = (m.stack()
            .rename("value")
            .reset_index())
    long.columns = ["origin", "destination", "value"]
    long["value"] = pd.to_numeric(long["value"], errors="coerce")*(-1 if neg_magnitudes else 1)
    long = long.dropna(subset=["value"])

    long = long[long["value"] > 0]
    return long.sort_values("value", ascending=False).reset_index(drop=True)


def sankey_from_matrix(
        flows_matrix: pd.DataFrame,
        days: int = 30,
        value_label: str = "Tonnes",
        value_scale: float = 1e6,
        scale_label: str = "M",
        title: str | None = None,
        region_or_country: str = "",
        counting_vessels: bool = False,
) -> go.Figure:
    """Build a Plotly Sankey from a wide origin x destination matrix.

    Parameters
    ----------
    flows_matrix : DataFrame indexed by origin region with destination columns.
    days         : window size, used only for the title text.
    value_label  : unit shown on labels and hover (e.g. 'tonnes', 'cm').
    value_scale  : divisor for the displayed totals (1e6 = millions).
    scale_label  : prefix shown next to the unit ('M' for millions).
    title        : override the chart title; default builds one automatically.
    """
    _decimals_to_show = 0 if counting_vessels else 1
    flows_matrix.columns = [c + "." for c in flows_matrix.columns]
    flows = matrix_to_flows(flows_matrix, neg_magnitudes=not counting_vessels)
    if flows.empty:
        raise ValueError("Flow matrix is empty after cleaning — no flows to plot.")

    # Order origins/destinations by total volume (largest first)
    origin_totals = flows.groupby("origin")["value"].sum().sort_values(ascending=False)
    dest_totals = flows.groupby("destination")["value"].sum().sort_values(ascending=False)
    origins = list(origin_totals.index)
    dests = list(dest_totals.index)

    # Build node list and lookup
    node_labels = (
            [f"<b>{o}</b> {origin_totals[o] / value_scale:.{_decimals_to_show}f} {scale_label} {value_label}"
             for o in origins] +
            [f"{dest_totals[d] / value_scale:.{_decimals_to_show}f} {scale_label} {value_label} <b>{d}</b>"
             for d in dests]
    )
    idx = {**{o: i for i, o in enumerate(origins)},
           **{d: len(origins) + i for i, d in enumerate(dests)}}

    # Color origins from the palette; destinations stay neutral gray
    origin_colors = {o: ORIGIN_PALETTE[i % len(ORIGIN_PALETTE)]
                     for i, o in enumerate(origins)}
    node_colors = [origin_colors[o] for o in origins] + ["#8A8A8A"] * len(dests)

    link_source = [idx[o] for o in flows["origin"]]
    link_target = [idx[d] for d in flows["destination"]]
    link_value = flows["value"].tolist()
    link_color = [hex_to_rgba(origin_colors[o], 0.55) for o in flows["origin"]]
    link_hover = [
        f"{o} -> {d}: {v / value_scale:.{_decimals_to_show}f} {scale_label} {value_label}"
        for o, d, v in zip(flows["origin"], flows["destination"], flows["value"])
    ]

    def stack_positions(node_list, totals):
        total = totals.sum()
        # gap between adjacent nodes, expressed as a fraction of column height
        gap_frac = 0.02
        n_gaps = max(len(node_list) - 1, 1)
        usable = 1.0 - n_gaps * gap_frac
        positions = {}
        cursor = 0.0
        for name in node_list:  # node_list is already sorted big -> small
            share = (totals[name] / total) * usable
            positions[name] = cursor + share / 2  # y is the node's center
            cursor += share + gap_frac
        return positions

    origin_y = stack_positions(origins, origin_totals)
    dest_y = stack_positions(dests, dest_totals)

    node_x = [0.001] * len(origins) + [0.999] * len(dests)
    node_y = [origin_y[o] for o in origins] + [dest_y[d] for d in dests]



    fig = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=40,
            thickness=18,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=link_source,
            target=link_target,
            value=link_value,
            color=link_color,
            customdata=link_hover,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))

    if title is None:
        today = pd.Timestamp.today().normalize().date()
        start = today - pd.Timedelta(days=days)
        title = (f"<b>LNG flows by {region_or_country} - rolling {days} days</b>"
                 f"<br><span style='font-size:13px;color:#666'>"
                 f"Delivered within {start} and {today}</span>")

    fig.update_layout(
        **_dark_layout(),
        height=900,  # default is ~450; bump this up
        width=1100,
        title=dict(text=title, x=0.01, xanchor="left"),
        # font=dict(family="Inter, Arial, sans-serif", size=13, color="#222"),
        # paper_bgcolor="white",
        # plot_bgcolor="white",
        # height=None,
        margin=dict(l=20, r=20, t=90, b=30),
    )

    return fig


def _dark_layout(**overrides):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(family="sans-serif", color="#C9D1D9", size=14),
        xaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
        yaxis=dict(gridcolor="#30363D", linecolor="#30363D"),
    )
    layout.update(overrides)
    return layout


def get_bsrch_lng_figs(save_plots=False):
    bapi_latest_lng_journeys = "584VC010OJ0QJK741S7PLC6YG"

    bikey_exported_lng = "24JY3973TCXRKFAVYPK9NRS3B"
    bikey_reexported_lng = "5UUQ5VLTLI5SK782KTRAIR6GQ"

    bikey_imported_lng = "CDFFYQDO5ZXIZLGLIE4UGUQOY"
    bikey_importedpartial_lng = "ED0X2R8L2MD2KOBFXKPFZEPUR"


    today = datetime.today().date()
    imported_lng = blp.bsrch("TPD:DEX",
                       {"BIKEY": bikey_imported_lng})
    exported_lng = blp.bsrch("TPD:DEX",
                       {"BIKEY": bikey_exported_lng})

    # in_transit_lng = exported_lng[exported_lng["Destination_Berthcountry"]=="In Transit"]

    imports_summary_last_n_days, od_matrix_last_n_days, transit_more_than_n_days = build_recap(
        imported_lng,
        exported_lng
    )

    df_in_transit = transit_more_than_n_days[1].join(transit_more_than_n_days[20], lsuffix="_0days", rsuffix="_20days").fillna(0)
    df_in_transit = df_in_transit.join(transit_more_than_n_days[30], rsuffix="_30days").fillna(0)

    df_in_transit.rename(columns={
        "Mtonnes_0days": "MillionMT_0days",
        "Mtonnes_20days": "MillionMT_20days",
        "Mtonnes": "MillionMT_30days",
        "Cargoes": "Cargoes_30days"
    }, inplace=True)

    cols_to_drop = [c for c in df_in_transit.columns if "Mcm" in c]

    df_in_transit.drop(columns=cols_to_drop, inplace=True)

    for c in ["MillionMT_0days", "MillionMT_20days", "MillionMT_30days"]:
        df_in_transit[c] = df_in_transit[c]/1000000

    lng_on_water_table_html = run_lng_on_water_processing(df_in_transit)

    _, od_matrix_last_n_days_region, _ = build_recap(
        imported_lng,
        exported_lng,
        od_index="Origin_Berthregion",
        aggfunc="count"
    )

    _fig_country = sankey_from_matrix(
            od_matrix_last_n_days[30],
            days = 30,
            value_label = "",
            value_scale = 1e6,
            scale_label = "Million MT",
            title = None,
            region_or_country="export country"
        )

    _fig_region = sankey_from_matrix(
        od_matrix_last_n_days_region[30],
        days=30,
        value_label="",
        value_scale=1,
        scale_label="Vessels",
        title=None,
        region_or_country="export region (Vessel Count)",
        counting_vessels=True,
    )


    if save_plots:
        out = f"C:\\Marti\\test_sankey_country_granular.html"
        _fig_country.write_html(out, include_plotlyjs=True, full_html=True)
        print(f"Saved Sankey to {out}")

        out = f"C:\\Marti\\test_sankey_region_granular.html"
        _fig_region.write_html(out, include_plotlyjs=True, full_html=True)
        print(f"Saved Sankey to {out}")

        with open('C:\\Marti\\lng_on_water.html', 'w', encoding='utf-8') as f:
            f.write(lng_on_water_table_html)

        print("Saved to C:\\Marti\\lng_on_water.html")

    return _fig_country, _fig_region, lng_on_water_table_html

if __name__ == "__main__":
    fig_country, _fig_region, html_table = get_bsrch_lng_figs(save_plots=True)






