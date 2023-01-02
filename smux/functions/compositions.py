import numpy as np

from .abstract_function import AbstactFunction


__all__ = ("Linearity", "TimeShift", "FreqShift", "TimeScaling")


class Linearity(AbstactFunction):
    def __init__(self, f, g=None, a: float = 1.0, b: float = 1.0) -> None:
        super().__init__()

        self.f = f
        self.g = g
        self.a = a
        self.b = b

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        if self.g is not None:
            return self.a * self.f(x) + self.b * self.g(x)
        return self.a * self.f(x)

    def ft(self, f):

        if self.g is not None:
            return self.a * self.f.ft(f) + self.b * self.g.ft(f)
        return self.a * self.f.ft(f)


class TimeShift(AbstactFunction):
    def __init__(self, f, tau: float) -> None:
        super().__init__()

        self.f = f
        self.tau = tau

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):

        return self.f(x - self.tau)

    def ft(self, f):
        
        return np.exp(-2j * np.pi * f * self.tau) * self.f.ft(f)

    # ========================================================================
    # Public interface
    # ========================================================================

    @property
    def diff(self):

        return TimeShift(self.f.diff, self.tau)


class FreqShift(AbstactFunction):
    def __init__(self, f, a: float) -> None:
        super().__init__()

        self.f = f
        self.a = a

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):
        return np.exp(1j * self.a * x) * self.f(x)

    def ft(self, f):
        return self.f.ft(f - self.a / (2 * np.pi))


class TimeScaling(AbstactFunction):
    def __init__(self, f, a: float) -> None:
        super().__init__()

        self.f = f
        self.a = a

    # ========================================================================
    # Implement AbstactFunction
    # ========================================================================

    def eval(self, x):
        return self.f(self.a * x)

    def ft(self, f):
        return 1.0 / np.abs(self.a) * self.f.ft(f / self.a)
