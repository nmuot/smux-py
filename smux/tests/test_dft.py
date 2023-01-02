import unittest

import numpy as np
import matplotlib.pyplot as plt
from smux.dft import dtft, idtft, dtft_f_max, dtft_f_min


class TestDFT(unittest.TestCase):
    def test_utils(self):

        sr = 256  # sampling rate
        ts = 1.0 / sr  # time step
        t = np.arange(0, 1, ts)

        self.assertAlmostEqual(dtft_f_max(t), 128)
        self.assertAlmostEqual(dtft_f_min(t), 1)

    def test_dtft(self):

        sr = 256  # sampling rate
        ts = 1.0 / sr  # time step
        t = np.arange(0, 1, ts)
        sp = np.sin(2 * np.pi * 60 * t)

        rfft = np.fft.rfft(sp) * t[1]
        freq = np.fft.rfftfreq(t.shape[-1], t[1])

        dft = dtft(sp, t, freq)

        np.testing.assert_allclose(dft.imag, rfft.imag, atol=1e-14)
        np.testing.assert_allclose(dft.real, 0, atol=1e-14)

    def test_idtft(self):

        sr = 256  # sampling rate
        ts = 1.0 / sr  # time step
        t = np.arange(0, 1, ts)
        sp = np.sin(2 * np.pi * 60 * t)

        freq = np.fft.rfftfreq(t.shape[-1], t[1])
        dft = np.zeros_like(freq, dtype=complex)
        dft[60] = -0.5j

        st = idtft(dft, freq, t)

        np.testing.assert_allclose(st, sp, atol=1e-13)
