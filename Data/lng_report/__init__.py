"""BGN LNG daily report package.

Cleaned-up, modular version of the original ``Data/exampe_test.py``:

- ``config``     : ticker maps, Bloomberg month codes, FX tenor grid.
- ``styling``    : dark Plotly layout + HTML page templates.
- ``bloomberg``  : ``BloombergExtractor`` and history pulls (xbbg).
- ``spark``      : ``extract_spark_quotes`` (Spark Commodities API).
- ``figures``    : the Plotly figures that make up the report.
- ``report``     : ``create_bgn_lng_report_grid`` orchestrator + entrypoint.
- ``emailer``    : Outlook email delivery.
- ``options_oi`` : standalone options open-interest analysis.
- ``violin``     : standalone TTF-NBP spread violin plot.

Submodules are imported explicitly (e.g. ``from Data.lng_report.report import
create_bgn_lng_report_grid``) so importing the package never drags in the
Bloomberg/Outlook stack unless you need it.
"""
