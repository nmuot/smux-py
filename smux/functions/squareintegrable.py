import numpy as np

from .abstract_function import AbstactFunction


__all__ = ("Rect", "Sinc", "SincSquare", "Tri", "CausalExp", "Gaussian")


class SquareIntegrableFunction(AbstactFunction):
    """The Square Integrable Function"""

    def __init__(self, a: float = 1.0) -> None:
        super().__init__()

        self.a = a


def rect(x):
    """usual rect function
    """

    return np.where(np.abs(x) <= 0.5, 1, 0)


def tri(x):
    """usual tri function
    """

    return np.max((1.0 - np.abs(x), np.zeros_like(x)), axis=0)


def heaviside(x):
    """usuale heaviside function.
    """

    return np.where(x > 0.0, 1.0, 0.0)


class Rect(SquareIntegrableFunction):

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return rect(self.a * x)

    def ft(self, f):

        return 1.0 / np.abs(self.a) * np.sinc(f / self.a)


class Sinc(SquareIntegrableFunction):

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return np.sinc(self.a * x)

    def ft(self, f):

        return 1.0 / np.abs(self.a) * rect(f / self.a)


class SincSquare(SquareIntegrableFunction):

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return np.sinc(self.a * x)**2

    def ft(self, f):

        return 1.0 / np.abs(self.a) * tri(f / self.a)


class Tri(SquareIntegrableFunction):

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return tri(self.a * x)

    def ft(self, f):

        return 1.0 / np.abs(self.a) * np.sinc(f / self.a) ** 2


class CausalExp(SquareIntegrableFunction):

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        ax = self.a * x
        return heaviside(ax) * np.exp(-ax)

    def ft(self, f):

        return 1.0 / (self.a + 2.0j * np.pi * f)


class Gaussian(SquareIntegrableFunction):

    # ========================================================================
    # Public interface
    # ========================================================================

    @property
    def diff(self):

        return DGaussian(self.a)

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return np.exp(-self.a * x**2)

    def ft(self, f):

        return np.sqrt(np.pi / self.a) * np.exp(-np.pi**2 * f**2 / self.a)


class DGaussian(SquareIntegrableFunction):

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return -2 * self.a * x * np.exp(-self.a * x**2)

    def ft(self, f):

        return (
            2j
            * np.pi
            * f
            * np.sqrt(np.pi / self.a)
            * np.exp(-np.pi**2 * f**2 / self.a)
        )
