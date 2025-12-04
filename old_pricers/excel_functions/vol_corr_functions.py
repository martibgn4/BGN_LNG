import xlwings as xw

from thorn.api import (
    build_term_vol, build_term_corr, get_spot_vol, SpotVolRatios, MeanRevertingProcesses, build_term_corr_matrix
)
from thorn_excel.utils.cache import CacheConverter, cache_results
from thorn_excel.utils.converter import DateConverter, TenorConverter, RemoveBlanks
from thorn_excel.utils.data_loader import load_underlyings
from thorn_excel.utils.docstring_utils import docstring_formatter
from thorn_excel.utils.replay_logger_utils import replay_logger

__all__ = [
    "QABuildVolCorr",
    "QABuildSpotVolRatios",
    "QATermVol",
    "QATermCorr",
    "QATermCorrMatrix",
    "QASpotVol",
]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('data', RemoveBlanks, ndim=2, doc='List of Lists containing the VolCorr data set as per the Calibration output')
@xw.arg('cache_label', doc='User defined name used for storage and retrieval of values.')
@xw.ret(doc="A label to be used to reference the built VolCorr object")
@load_underlyings
@replay_logger()
@cache_results("VolCorr")
def QABuildVolCorr(data, cache_label=None):
    """Builds VolCorr object from VolCorr calibration results."""
    return MeanRevertingProcesses.from_table(data)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('data', RemoveBlanks, ndim=2, doc='List of Lists containing the SpotVolRatios data set')
@xw.arg('cache_label', doc='User defined name used for storage and retrieval of values.')
@xw.ret(doc="A label to be used to reference the built SpotVolRatios object")
@load_underlyings
@replay_logger()
@cache_results("SpotVolRatios")
def QABuildSpotVolRatios(data, cache_label=None):
    """Builds SpotVolRatios object from spot vol ratio calibration results."""
    return SpotVolRatios.from_table(data)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc='Pricing Date')
@xw.arg('vol_corr', CacheConverter, doc='Handle to VolCorr object')
@xw.arg('underlying', doc="An underlying name which exists within the VolCorr object")
@xw.arg('delivery_period', TenorConverter, doc=" Any forward delivery period")
@xw.arg('expiry_date', DateConverter, doc="[Optional] Option expiry date")
@xw.arg('fwd_curve', CacheConverter, doc="[Optional] Forward Curve (required for Peak / Offpeak underlyings)")
@xw.arg('spot_vol_ratios', CacheConverter, doc="[Optional] Spot Vol Ratios")
@xw.arg('pillared_smiles_collection', CacheConverter,
        doc='[Optional] Handle to QAPillaredSmileCollection object, used to calculate implied vol corr.')
@xw.ret(doc="A terminal volatility for the given underlying and delivery period")
@load_underlyings
@replay_logger()
def QATermVol(
    pricing_date, vol_corr, underlying, delivery_period, expiry_date=None, fwd_curve=None,
    spot_vol_ratios=None, pillared_smiles_collection=None
):
    """
    Get the terminal volatility for the given underlying and delivery period.
    Uses ImpliedVolCorr if pillared_smiles_collection is provided, VolCorr otherwise
    """
    return build_term_vol(
        vol_corr, pricing_date, underlying, [delivery_period],
        expiry_date, fwd_curve, spot_vol_ratios, pillared_smiles_collection
    )[0]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc='Pricing Date')
@xw.arg('vol_corr', CacheConverter, doc='Handle to VolCorr object')
@xw.arg('underlying1', doc="An underlying name which exists within the VolCorr object")
@xw.arg('delivery_period1', TenorConverter, doc=" Any forward delivery period")
@xw.arg('underlying2', doc="An underlying name which exists within the VolCorr object")
@xw.arg('delivery_period2', TenorConverter, doc=" Any forward delivery period")
@xw.arg('expiry_date', DateConverter, doc="[Optional] Option expiry date")
@xw.ret(doc="A terminal correlation between two underlyings at their delivery periods")
@load_underlyings
@replay_logger()
def QATermCorr(pricing_date, vol_corr, underlying1, delivery_period1, underlying2, delivery_period2, expiry_date=None):
    """Get the terminal correlation between two underlyings at their delivery periods."""
    return build_term_corr(
        vol_corr, pricing_date, underlying1, [delivery_period1], underlying2, [delivery_period2], expiry_date
    )[0]


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc='Pricing Date')
@xw.arg('vol_corr', CacheConverter, doc='Handle to VolCorr object')
@xw.arg('underlying1', doc="An underlying name which exists within the VolCorr object")
@xw.arg('delivery_periods', TenorConverter, doc="List of forward delivery periods")
@xw.arg('underlying2', doc="[Optional] An underlying name which exists within the VolCorr object (default=underlying1)")
@xw.arg('expiry_date', DateConverter, doc="[Optional] Option expiry date")
@xw.ret(doc="A terminal correlation between two underlyings at their delivery periods")
@load_underlyings
@replay_logger()
def QATermCorrMatrix(pricing_date, vol_corr, underlying1, delivery_periods, underlying2=None, expiry_date=None):
    """Get the terminal correlation between two underlyings at their delivery periods."""
    return build_term_corr_matrix(vol_corr, pricing_date, underlying1, delivery_periods, underlying2, expiry_date)


@xw.func(call_in_wizard=False)
@docstring_formatter
@xw.arg('pricing_date', DateConverter, doc='Pricing Date')
@xw.arg('vol_corr', CacheConverter, doc='Handle to VolCorr object')
@xw.arg('underlying', doc="An underlying name which exists within the VolCorr object")
@xw.arg('ccy', doc='IR Pivot Currency, Needed if a pillared_smiles_collection is supplied')
@xw.arg('pillared_smiles_collection', CacheConverter,
        doc='[Optional] Handle to QAPillaredSmileCollection object, used to calculate implied vol corr.')
@xw.ret(doc="Spot Volatility")
@load_underlyings
@replay_logger()
def QASpotVol(pricing_date, vol_corr, underlying, ccy=None, pillared_smiles_collection=None):
    """Get the Spot volatility for an underlying."""
    return get_spot_vol(vol_corr, pricing_date, underlying, ccy, pillared_smiles_collection)


