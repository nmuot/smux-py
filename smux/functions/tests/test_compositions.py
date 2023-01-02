import unittest

import numpy as np
import smux.functions.squareintegrable as funcs
import smux.functions.compositions as comp


class TestCompositionFunction(unittest.TestCase):
    def test_linearity(self):

        f = funcs.Gaussian()
        g = funcs.DGaussian()
        func = comp.Linearity(f, g, 1, 2)

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func(x), f(x) + 2 * g(x))

        y = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func.ft(y), f.ft(y) + 2 * g.ft(y))

        func = comp.Linearity(f, a=2)

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func(x), 2 * f(x))

        y = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func.ft(y), 2 * f.ft(y))

    def test_timeshift(self):

        g = funcs.Gaussian()
        func = comp.TimeShift(g, 1)

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func(x), g(x - 1))

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func.ft(f), np.exp(-2j * np.pi * f) * g.ft(f))

        diff = func.diff
        self.assertIsInstance(diff, comp.TimeShift)
        self.assertIsInstance(diff.f, funcs.DGaussian)
        self.assertAlmostEqual(diff.f.a, func.f.a)
        self.assertAlmostEqual(diff.tau, func.tau)

    def test_freqshift(self):

        g = funcs.Gaussian()
        func = comp.FreqShift(g, 1)

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func(x), g(x) * np.exp(1j * x))

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func.ft(f), g.ft(f - 0.5 / np.pi))

    def test_timescaling(self):

        g = funcs.Gaussian()
        func = comp.TimeScaling(g, 2)

        x = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func(x), g(2 * x))

        f = np.linspace(-2, 2, 11)
        np.testing.assert_allclose(func.ft(f), 0.5 * g.ft(0.5 * f))
