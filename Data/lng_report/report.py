"""Assemble the BGN LNG daily report: build every figure, render the HTML
(interactive grid + emailed image table), save it and send it by email.

Run as a script to generate and email today's report:
    python -m Data.lng_report.report
"""

import base64
from datetime import datetime

import plotly.io as pio

from Data.bloomberg_bsrch import get_bsrch_lng_figs

from Data.lng_report.emailer import send_email
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

# Where the finished report is written: a local copy, a dated shared-drive copy
# and a stable "latest" copy on the shared drive.
_ONLINE_DIR = ("C:\\Users\\marti.fernandezreal\\BAYEGAN DIS TIC. A.S\\"
               "LNG Team - 01. Miscellaneous\\17. LNG BGN Reports")


def _fig_to_base64_img(fig, width=600, height=400):
    """Convert a Plotly figure to a base64-embedded <img> tag for email."""
    img_bytes = fig.to_image(format="png", width=width, height=height)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:{width}px; display:block;">'


def create_bgn_lng_report_grid(report_date):
    """Build all report figures, save the interactive HTML and email it."""
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

    # Interactive HTML report (Plotly divs).
    divs = [pio.to_html(f, full_html=False, include_plotlyjs=False) for f in figs_to_htmlise]
    divs = divs + [html10]
    html = build_grid_report_html(report_date, divs)

    out_path_online = f"{_ONLINE_DIR}\\lng_report_{pricing_date}.html"
    out_path_online_base = f"{_ONLINE_DIR}\\lng_report.html"
    for save_path in [out_path_online, out_path_online_base]:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Interactive report saved to {save_path}")

    # Static image version for the email body.
    img_tags = [_fig_to_base64_img(fig) for fig in figs_to_htmlise] + [html10]
    simplified_html = build_email_html(report_date, img_tags)

    send_email(
        simplified_html,
        html_out_path=out_path_online,
        sending_to="lng@bgn-int.com; vasileios.giannoutsos@bgn-int.com"
        # sending_to="marti.fernandezreal@bgn-int.com; vasileios.giannoutsos@bgn-int.com"
    )


def main():
    pricing_date = datetime.now().date()
    create_bgn_lng_report_grid(pricing_date)


if __name__ == "__main__":
    main()
