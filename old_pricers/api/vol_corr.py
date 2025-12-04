from quantity import DAYS_PER_YEAR
from thorn.core.market.dynamic.ivc_mkt_dynamic import IvcMktDynamic
from thorn.core.market.dynamic.mkt_dynamic_builder import MktDynamicBuilder
from thorn.core.market.dynamic.vc_mkt_dynamic import VcMktDynamic
from thorn.core.market.forward.discount_curve import NullDiscountCurve
from thorn.core.market.mkt_env.keys import (
    IrPivotCurrency, PricingDate, CurveDate, MeanRevertingProcessesKey, VolSurfaceKey, MktEnvKey, SpotVolRatiosKey,
)
from thorn.core.market.mkt_env.mkt_env import MktEnv
from thorn.core.market.volatility.pillared_smile_collection import ParameterisedPillaredSmileCollectionFactory
from thorn.core.market.volatility.vc.mean_reverting_processes import MeanRevertingProcesses
from thorn.core.market.volatility.vc.vol_corr import SpotVolRatios, VolCorr
from thorn.core.market.volatility.vc.vol_corr_adaptor import DatedVolCorrAdaptor
from underlying import SwapRate, Underlying

import pandas as pd


__all__ = [
    "ParameterisedPillaredSmileCollectionFactory",
    "MeanRevertingProcesses",
    "SpotVolRatios",
    "build_term_vol",
    "build_term_corr",
    "build_term_corr_matrix",
    "get_spot_vol",
]


_ = ParameterisedPillaredSmileCollectionFactory
_ = MeanRevertingProcesses
_ = SpotVolRatios


def build_term_vol(
    mean_rev_procs, pricing_date, underlying, delivery_periods, expiry_date=None,
    fwd_curve=None, spot_vol_ratios=None, pillared_smiles_collection=None
):
    """
    Builds a minimal MktEnv and VcMktDynamic (or IvcMktDynamic) so that get_volatility can be used to compute vols for
    baskets as well as directly modelled underlyings
    """
    underlying = Underlying.parse(underlying)
    quotes = {
        PricingDate: pricing_date,
        CurveDate: pricing_date,
        MeanRevertingProcessesKey: mean_rev_procs,
        IrPivotCurrency: underlying.currency,
        underlying.currency: NullDiscountCurve()
    }

    if pillared_smiles_collection:
        quotes[VolSurfaceKey(underlying)] = pillared_smiles_collection
    if fwd_curve:
        quotes[underlying] = fwd_curve
    if spot_vol_ratios:
        quotes[SpotVolRatiosKey] = spot_vol_ratios

    static_data = {
        MktEnvKey: MktEnv(quotes),
    }
    
    mkt_dynamic_cls = IvcMktDynamic if pillared_smiles_collection else VcMktDynamic
    mkt_dynamic = MktDynamicBuilder(mkt_dynamic_cls)(static_data)
    term_vols = [
        mkt_dynamic.get_volatility(underlying, tp, expiry_date, strike=None)
        for tp in delivery_periods
    ]
    return term_vols


def build_term_corr(mean_rev_procs, pricing_date, underlying1, delivery_periods1, underlying2,
                    delivery_periods2, expiry_date=None):
    vol_corr_adaptor = DatedVolCorrAdaptor(VolCorr(mean_rev_procs), pricing_date)
    term_corr = [
        vol_corr_adaptor.calc_corr(SwapRate(underlying1, dp1), SwapRate(underlying2, dp2), ex_date=expiry_date)
        for dp1, dp2 in zip(delivery_periods1, delivery_periods2)
    ]
    return term_corr


def build_term_corr_matrix(mean_rev_procs, pricing_date, underlying1, delivery_periods,
                           underlying2=None, expiry_date=None):
    underlying2 = underlying2 if underlying2 else underlying1
    vol_corr_adaptor = DatedVolCorrAdaptor(VolCorr(mean_rev_procs), pricing_date)
    m = {
        dp1: [
            vol_corr_adaptor.calc_corr(SwapRate(underlying1, dp1), SwapRate(underlying2, dp2), ex_date=expiry_date)
            for dp2 in delivery_periods
        ] for dp1 in delivery_periods
    }

    df = pd.DataFrame.from_dict(m, orient='index', columns=delivery_periods)
    return df.rename(index=lambda s: f"{underlying1}:{s}", columns=lambda s: f"{underlying2}:{s}")


def get_spot_vol(mean_rev_procs, pricing_date, underlying, currency=None, pillared_smiles_collection=None):
    if isinstance(underlying, str):
        underlying = Underlying.parse(underlying)
    if pillared_smiles_collection:
        quotes = {
            IrPivotCurrency: currency,
            PricingDate: pricing_date,
            CurveDate: pricing_date,
            MeanRevertingProcessesKey: mean_rev_procs,
            VolSurfaceKey(underlying): pillared_smiles_collection
        }
        ivc_mkt_dyn = IvcMktDynamic(MktEnv(quotes), {})
        vol_corr = ivc_mkt_dyn.vol_corr
    else:
        vol_corr = VolCorr(mean_rev_procs)
    delivery_start = 0.0 / DAYS_PER_YEAR
    delivery_length = 1.0 / DAYS_PER_YEAR
    spot_vol = vol_corr.calc_inst_vol(underlying, delivery_start, delivery_length)
    return spot_vol


