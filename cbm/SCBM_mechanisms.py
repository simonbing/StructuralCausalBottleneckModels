from abc import ABC, abstractmethod
import inspect

import numpy as np

from cbm import make_iterable, get_cond_mean_cov


class SCBMMechanism(ABC):
    """
    Abstract base class for SCBM mechanisms.
    """
    @abstractmethod
    def __init__(self):
        self.mechanism = self._get_mechanism()
        return NotImplementedError

    @abstractmethod
    def _get_mechanism(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.mechanism(*args, **kwargs)

    # This method will modify the mechanism property
    @abstractmethod
    def intervene(self, target, value):
        """
        Args:
            target: list[int]
                indices of microstates that are targeted by an intervention
            values: list
                list of values that the targeted microstates are set to
        """
        raise NotImplementedError


class GaussianLangevinMechanism(SCBMMechanism):
    """
    Mechanism class that implements the Langevin diffusion dynamics for the
    multivariate Gaussian case.

    Attributes:
        mu: vector or function
            mean vector
        E: np.array
            covariance matrix
    """
    def __init__(self, mu, E):
        if not inspect.isfunction(mu):

            def mu_f():
                return mu
            # mu_f = lambda : mu
            self.mu = mu_f
        else:
            self.mu = mu
        self.E = E

        super().__init__()

    def _get_mechanism(self):
        # Sample from a multivariate Gaussian
        L = np.linalg.cholesky(self.E)

        def f(noise, *args):
            return self.mu(*args) + (L @ noise.T).T
        return f

    def intervene(self, target, value):
        value = make_iterable(value)

        mu_bar, E_bar = get_cond_mean_cov(target, value, self.E)
        L_bar = np.linalg.cholesky(E_bar)

        def f(noise, *args):
            # Allocate array for output
            out = np.empty_like(noise)
            # Apply mu mechanism and drop intervened dims we don't need
            mu_mech = self.mu(*args)
            mu_mech = np.delete(mu_mech, target, axis=1)
            # Get correct size of noise
            noise = np.delete(noise, target, axis=1)
            sample = mu_mech + (mu_bar + (L_bar @ noise.T).T)
            # Populate out array at correct indices
            target_count = 0
            sample_count = 0
            for i in range(out.shape[1]):
                if i in target:
                    out[:, i] = value[target_count] * np.ones_like(out[:, i], dtype=float)
                    target_count += 1
                else:
                    out[:, i] = sample[:, sample_count]
                    sample_count += 1
            return out

        self.mechanism = f