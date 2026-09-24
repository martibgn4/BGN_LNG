"""Assemble the BGN LNG daily report: build every figure, render the HTML
(interactive grid + emailed image table), save it and send it by email.

Run as a script to generate and email today's report:
    python -m Data.lng_report.report
"""

import base64
import io
from datetime import datetime

import matplotlib.pyplot as plt
import plotly.io as pio

from Data.bloomberg_bsrch import get_bsrch_lng_figs
from Data.lng_report.bloomberg import get_historical_prices_for_underlyings

from Data.lng_report.config import ONLINE_DIR
from Data.lng_report.emailer import send_email
from Data.lng_report.platts import get_platts_flow_figs
from Data.lng_report.styling import build_grid_report_html, build_email_html
from Data.lng_report.figures import (
    plot_spark_quotes,
    plot_nicely_historical_prices,
    plot_ttf_and_eu_spreads,
    plot_hh_tfu_jkm,
    plot_nicely_lng_us_flows,
    plot_nicely_baltic_freight,
    plot_forwards_baltic_freight,
    plot_nicely_lng_on_water_curves,
)

# Where the finished report is written: a dated shared-drive copy and a stable
# "latest" copy. ONLINE_DIR itself lives in ``config`` so the settlement report
# can publish alongside without importing the whole plotting stack.

DEFAULT_RECIPIENTS = ("lng@bgn-int.com; vasileios.giannoutsos@bgn-int.com; "
                      "ahmed.shhati@bgn-int.com")


def _fig_to_base64_img(fig, width=600, height=400):
    """Convert a Plotly figure to a base64-embedded <img> tag for email."""
    img_bytes = fig.to_image(format="png", width=width, height=height)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:{width}px; display:block;">'


def _mpl_fig_to_base64_img(fig, dpi=110, max_width=1400):
    """Convert a matplotlib figure to a base64-embedded <img> tag and close it.

    The same tag is reused by the interactive page and the email — these
    figures are static either way, so there is nothing to gain from a second
    render.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%; max-width:{max_width}px; display:block; '
            f'margin:0 auto;">')


def _platts_flow_img_tags():
    """MOC counterparty-flow charts, or an empty list if Platts is unreachable.

    The daily report must still go out when the Platts API is down, so a
    failure here is logged and skipped rather than raised.
    """
    try:
        figs = get_platts_flow_figs(contract=None, cumulative=True, dark=True)
    except Exception as exc:
        print(f"Platts MOC flow charts skipped: {exc!r}")
        return []
    return [_mpl_fig_to_base64_img(fig) for fig in figs]


def create_bgn_lng_report_grid(report_date, sending_to=DEFAULT_RECIPIENTS, send=True):
    """Build all report figures, save the interactive HTML and email it.

    Set `send=False` to build and save without mailing -- useful for a dry run
    of the daily job.
    """
    pricing_date = datetime.now().date()

    fig1 = plot_spark_quotes()
    fig2 = plot_nicely_historical_prices(
        ttf_tick="TYRF27 Comdty", jkm_tick="AJKMY 27 BCFV Index",
        hh_tick="FFHHY 27 Index", brent_tick="FSBTY 27 Index",
        month="Cal27 Contracts", back_periods=180
    )
    fig3 = plot_ttf_and_eu_spreads(years=3, today_date=pricing_date, plot_spreads=False)
    fig4 = plot_hh_tfu_jkm()
    fig5 = plot_nicely_lng_us_flows(back_periods=360)
    fig6 = plot_nicely_baltic_freight(
        blng1_ticker="IKA1 Comdty",
        blng2_ticker="IKD1 Comdty",
        blng3_ticker="IKI1 Comdty",
        spark30_ticker="LBE1 Comdty",
        back_periods=720
    )
    fig7 = plot_forwards_baltic_freight()
    fig8, fig9, html10 = get_bsrch_lng_figs()
    fig11 = plot_nicely_lng_on_water_curves(
        lng_on_water_20d_count="LNGG20DC Index",
        lng_on_water_30d_count="LNGG30DC Index",
        lng_on_water_20d_vol="LNGG20DT Index",
        lng_on_water_30d_vol="LNGG30DT Index",
        back_periods=720
    )

    figs_to_htmlise = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig11]

    # Platts MOC flow (matplotlib): embedded as images and laid out full width.
    platts_img_tags = _platts_flow_img_tags()

    # Interactive HTML report (Plotly divs).
    divs = [pio.to_html(f, full_html=False, include_plotlyjs=False) for f in figs_to_htmlise]
    divs = divs + [html10]
    html = build_grid_report_html(report_date, divs, full_width_divs=platts_img_tags)

    out_path_online = f"{ONLINE_DIR}\\lng_report_{pricing_date}.html"
    out_path_online_base = f"{ONLINE_DIR}\\lng_report.html"
    for save_path in [out_path_online, out_path_online_base]:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Interactive report saved to {save_path}")

    # Static image version for the email body.
    img_tags = [_fig_to_base64_img(fig) for fig in figs_to_htmlise] + [html10]
    simplified_html = build_email_html(report_date, img_tags,
                                       full_width_img_tags=platts_img_tags)

    if not send:
        print("Dry run: LNG report built and saved, not emailed")
        return out_path_online

    send_email(
        simplified_html,
        subject=f"LNG Report - {report_date.isoformat()} ",
        sending_to=sending_to,
        attachment_path=out_path_online,
    )
    return out_path_online


def main(sending_to=DEFAULT_RECIPIENTS, send=True):
    pricing_date = datetime.now().date()
    create_bgn_lng_report_grid(pricing_date, sending_to=sending_to, send=send)


if __name__ == "__main__":

    # main()

    import pandas as pd
    dates = pd.date_range(end=datetime.today(), periods=900)

    fig2 = plot_nicely_historical_prices(
        ttf_tick="TYRF27 Comdty", jkm_tick="AJKMY 27 BCFV Index",
        hh_tick="FFHHY 27 Index", brent_tick="FSBTY 27 Index",
        month="Cal27 Contracts", back_periods=180
    )
    fig2.show()
    # df.to_csv("C:\\Marti\\historical_prices_2.csv")