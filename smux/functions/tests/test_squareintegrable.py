import unittest

import numpy as np
import smux.functions.squareintegrable as funcs


class TestSquareIntegrableFunction(unittest.TestCase):
    def test_rect(self):

        rect = funcs.Rect()

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(rect(x), (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0))

        f = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(rect.ft(f), np.sinc(f))

    def test_sinc(self):

        sinc = funcs.Sinc()

        x = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(sinc(x), np.sinc(x))

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(sinc.ft(f), (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0))

    def test_sinc_2(self):

        sinc = funcs.SincSquare()

        x = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(sinc(x), np.sinc(x) ** 2)

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(
            sinc.ft(f), (0, 0, 0, 0.2, 0.6, 1, 0.6, 0.2, 0, 0, 0)
        )

    def test_tri(self):
        tri = funcs.Tri()

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(tri(x), (0, 0, 0, 0.2, 0.6, 1, 0.6, 0.2, 0, 0, 0))

        f = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(tri.ft(f), np.sinc(f) ** 2)

    def test_uexp(self):

        uexp = funcs.CausalExp()

        x = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(
            uexp(x),
            np.array(
                [0, 0, 0, 0, 0]
                + [np.exp(-1 / 2), np.exp(-1), np.exp(-3 / 2), np.exp(-2)]
            ),
        )

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(uexp.ft(f), 1 / (1 + 2j * np.pi * f))

    def test_gauss(self):

        gauss = funcs.Gaussian()

        x = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(
            gauss(x),
            np.exp(-(x**2)),
        )

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(
            gauss.ft(f), np.sqrt(np.pi) * np.exp(-np.pi**2 * f**2)
        )

        self.assertIsInstance(gauss.diff, funcs.DGaussian)
        self.assertAlmostEqual(gauss.diff.a, gauss.a)

    def test_dgauss(self):

        dgauss = funcs.DGaussian()

        x = np.array((-2, -3 / 2, -1, -1 / 2, 0, 1 / 2, 1, 3 / 2, 2))
        np.testing.assert_allclose(
            dgauss(x),
            -2 * x * np.exp(-(x**2)),
        )

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(
            dgauss.ft(f), 2j * np.pi * f * np.sqrt(np.pi) * np.exp(-np.pi**2 * f**2)
        )
