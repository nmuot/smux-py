"""Implement `phyicale` discret Fourier Transforme.
"""

import numpy as np


def _d_x(x):
    "Eval trapz dx for FT"

    d_x = np.zeros_like(x)
    d_x[:-1] = x[1:] - x[:-1]
    d_x[-1] = d_x[-2]
    return d_x


def _dtft(a, x, y, ker=2 * np.pi):
    """Discrete Fourier Transforme 1D

    Args:
        a (1d-array): The intput array
        t (1d-array): The time array
        f (1d-array): The output frequencies array
        ker (float, optional): The integral kernel. Defaults to 2*pi.

    Returns: out
        complex 1d-array: The 1D DFT
    """

    assert len(a) == len(x)

    d_x = _d_x(x)
    k = np.expand_dims(y, -1) * x * ker
    e = d_x * np.exp(-1j * k)
    return np.dot(e, a)


def dtft_f_max(x):
    "Evaluate the maximum frequency accesible by DFT with given time sample."

    return np.min(1 / _d_x(x)) / 2


def dtft_f_min(x):
    "Evaluate the min frequency accesible by DFT with given time sample."

    return 1 / np.sum(_d_x(x))


def dtft(a, t, f):
    """Discrete Fourier Transforme 1D

    Args:
        a (1d-array): The intput array
        t (1d-array): The time array
        f (1d-array): The output frequencies array

    Returns: out
        complex 1d-array: The 1D DFT
    """

    return _dtft(a, t, f)


def idtft(a, f, t):
    """Inverse Discrete Fourier Transforme 1D

    Args:
        a (1d-array): The intput array
        f (1d-array): The frequencies array
        t (1d-array): The output time array

    Returns: out
        real 1d-array: The 1D DFT
    """

    return 2 * _dtft(a, f, t, ker=-2.0 * np.pi).real
