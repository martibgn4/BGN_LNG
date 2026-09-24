"""Forecasting model for the NWE LNG basis to TTF.

Sign convention, fixed in config/target.yaml and used everywhere:
``basis = DES_NWE - TTF`` in USD/MMBtu, so a negative basis is a discount.
"""

__all__ = ["config", "store", "features", "backtest"]
