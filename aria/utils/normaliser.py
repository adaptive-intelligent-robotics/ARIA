import abc


class Normaliser(abc.ABC):
    @abc.abstractmethod
    def apply(self, x):
        ...

    def __call__(self, x):
        return self.apply(x)

class NoNormaliser(Normaliser):
    def apply(self, x):
        return x

class NormaliserMinMax(Normaliser):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

        super().__init__()

    def apply(self, x):
        return (x - self.min_val) / (self.max_val - self.min_val)

    def __call__(self, x):
        return self.apply(x)
