import datetime
import pandas as pd
import unittest
from unittest.mock import call, patch

from thorn.api import (
    build_term_corr, build_term_vol, get_spot_vol, MeanRevertingProcesses, SpotVolRatios, build_term_corr_matrix
)
from thorn.api.test.context_fixtures import underlying_loader_context
from thorn_excel import CACHE, QABuildVolCorr, QASpotVol, QATermCorr, QATermVol, QABuildSpotVolRatios, QATermCorrMatrix
from thorn_excel.functions.test.excel_utils import xlwings_call_udf
from thorn_excel.functions.test.fixtures import psc_fixture, mean_rev_procs_fixture
from thorn_excel.utils.test.context_fixtures import cache_context


class TestVolCorrFunction:

    def setup_method(self):
        self.volcorr = mean_rev_procs_fixture()
        self.psc = psc_fixture()

    def setup_cache(self):
        CACHE.add_entry("", [], self.volcorr, "", "VolCorrHandle")
        CACHE.add_entry("", [], self.psc, "", "PSCHandle")

    @cache_context()
    @patch("thorn_excel.functions.vol_corr_functions.build_term_vol", autospec=build_term_vol, return_value=[6])
    @underlying_loader_context()
    def test_qa_term_vol(self, m_func):
        self.setup_cache()
        args = ('2018-01-01', "VolCorrHandle", "NBP", "2018_01")
        actual = xlwings_call_udf(QATermVol, args)[0][0]
        assert 6 == actual
        expected_args = (self.volcorr, datetime.date(2018, 1, 1), "NBP", ["2018_01"], None, None, None, None)
        assert call(*expected_args) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vol_corr_functions.build_term_corr", autospec=build_term_corr, return_value=[6])
    @underlying_loader_context()
    def test_qa_term_corr(self, m_func):
        self.setup_cache()
        args = ('2018-01-01', "VolCorrHandle", "NBP", "2018_01", "TTF", "2018_01")
        actual = xlwings_call_udf(QATermCorr, args)[0][0]
        assert 6 == actual
        expected_args = (self.volcorr, datetime.date(2018, 1, 1), "NBP", ["2018_01"], "TTF", ["2018_01"], None)
        assert call(*expected_args) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vol_corr_functions.build_term_corr_matrix", autospec=build_term_corr_matrix)
    @underlying_loader_context()
    def test_qa_term_corr_matrix(self, m_func):
        m_func.return_value = pd.DataFrame(
            data=[[0.95, 0.84], [0.86, 0.96]],
            index=["NBP:2018_01", "NBP:2018_02"], columns=["TTF:2018_01", "TTF:2018_02"]
        )
        expected = [
            [           "", "TTF:2018_01", "TTF:2018_02"],
            ["NBP:2018_01",          0.95,          0.84],
            ["NBP:2018_02",          0.86,          0.96],
        ]
        self.setup_cache()
        args = ('2018-01-01', "VolCorrHandle", "NBP", ["2018_01", "2018_02"], "TTF")
        actual = xlwings_call_udf(QATermCorrMatrix, args)
        assert expected == actual
        expected_args = (self.volcorr, datetime.date(2018, 1, 1), "NBP", ["2018_01", "2018_02"], "TTF", None)
        assert call(*expected_args) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vol_corr_functions.get_spot_vol", autospec=get_spot_vol, return_value=6)
    @underlying_loader_context()
    def test_qa_spot_vol(self, m_func):
        self.setup_cache()
        args = ('2018-01-01', "VolCorrHandle", "NBP", "GBP", "PSCHandle")
        actual = xlwings_call_udf(QASpotVol, args)[0][0]
        assert 6 == actual
        expected_args = (self.volcorr, datetime.date(2018, 1, 1), "NBP", "GBP", self.psc)
        assert call(*expected_args) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vol_corr_functions.MeanRevertingProcesses.from_table",
           autospec=MeanRevertingProcesses.from_table, return_value=7)
    @underlying_loader_context()
    def test_qa_build_vol_corr(self, m_func):
        vol_corr_2d = [[1, 2], [3, 4]]
        label = xlwings_call_udf(QABuildVolCorr, [vol_corr_2d])[0][0]
        actual = CACHE.lookup_by_label(label)
        assert 7 == actual
        assert call(vol_corr_2d) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vol_corr_functions.SpotVolRatios.from_table",
           autospec=SpotVolRatios.from_table, return_value=8)
    @underlying_loader_context()
    def test_qa_build_spot_vol_ratios(self, m_func):
        spot_vol_ratio_data = [[None, "NBP"], [1, 1.1], [2, 1.2]]
        label = xlwings_call_udf(QABuildSpotVolRatios, [spot_vol_ratio_data])[0][0]
        actual = CACHE.lookup_by_label(label)
        assert 8 == actual
        assert call(spot_vol_ratio_data) == m_func.call_args


