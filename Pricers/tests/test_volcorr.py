import unittest
import numpy as np

from BGN_LNG.Pricers.volcorr import VolCorr


class TestVolCorr(unittest.TestCase):
    def setUp(self):
        self.underlyings = ["TTF"]
        self.sigmas = [0.2, 0.3]
        self.alphas = [0, 1]
        self.corr_matrix = np.array([[1, 0.2], [0.2, 1]])

    def test_volcorr(self):
        volcorr_object = VolCorr(self.underlyings, self.sigmas, self.alphas, self.corr_matrix)

    def test_wrong_volcorr_definition(self):
        with self.assertRaisesRegex(AssertionError,"Length of sigmas should*"):
            _ = VolCorr(["TTF", "NBP"], self.sigmas, self.alphas, self.corr_matrix)

        with self.assertRaisesRegex(AssertionError, "Sigmas and Alphas must have*"):
            _ = VolCorr(self.underlyings, [0.1], self.alphas, self.corr_matrix)

        with self.assertRaisesRegex(AssertionError,
                               "Correlation matrix should be symmetric"):
            _ = VolCorr(self.underlyings, self.sigmas, self.alphas, np.array([[1, 0.3],[0.2, 1]]))
        with self.assertRaisesRegex(AssertionError, "Correlation matrix should have 1*"):
            _ = VolCorr(self.underlyings, self.sigmas, self.alphas, np.array([[1, 0.3],[0.3, 0.9]]))
