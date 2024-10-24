from abc import ABC, abstractmethod


class BaseRegressor(ABC):
    def __init__(self, seed, d_micro_in, d_micro_out, d_bottleneck, source,
                 target, d_cond=0):
        self.seed = seed
        self.d_micro_in = d_micro_in
        self.d_micro_out = d_micro_out
        self.d_bottleneck = d_bottleneck
        self.source = source
        self.target = target
        self.d_cond = d_cond

    @abstractmethod
    def fit(self, X, Y, X_cond=[]):
        raise NotImplementedError

    @abstractmethod
    def get_bottleneck_fct(self):
        """
        This should return a function that can be called to embed samples to the bottleneck space.
        """
        raise NotImplementedError
