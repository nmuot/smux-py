from abc import ABCMeta, abstractmethod

__all__ = ("AbstactFunction",)


class AbstactFunction(metaclass=ABCMeta):
    @abstractmethod
    def ft(self, f):
        """Evaluate the Fourier transforme

        Args:
            f (1d-array): The input array

        Returns:
            1d-array: The Fourier transforme
        """

    @abstractmethod
    def eval(self, x):
        """Evaluate the function at given x

        Args:
            x (1d-array): The input value

        Return y:
            1d-array: The output value
        """

    def __call__(self, x):

        return self.eval(x)
