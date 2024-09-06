from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from cbm import GaussianLangevinMechanism, MacroCausalVar, SCBM
from cbm.data import SCBMSampler
from cbm.data.utils import sample_mrf_prec
from cbm import make_iterable


class BaseRegressor(ABC):
    def __init__(self, d_micro_in, d_micro_out, d_bottleneck):
        self.d_micro_in = d_micro_in
        self.d_micro_out = d_micro_out
        self.d_bottleneck = d_bottleneck

    @abstractmethod
    def fit(self, X, Y, X_cond=[]):
        raise NotImplementedError

    @abstractmethod
    def get_bottleneck_fct(self):
        """
        This should return a function that can be called to embed samples to the bottleneck space.
        """
        raise NotImplementedError


class LinRegressor(BaseRegressor):
    def __init__(self, d_micro_in, d_micro_out, d_bottleneck):
        super().__init__(d_micro_in, d_micro_out, d_bottleneck)

        self.model = LinearRegression()

    def fit(self, X, Y, X_cond=[]):
        if len(X_cond) == 0:  # no conditioning
            self.model.fit(X, Y)
        else:
            self.X_dim = X.shape[-1]  # need this later in get_bottleneck_fct

            X_cat = np.concatenate((X, X_cond), axis=1)
            # print(f'X_cat: {X_cat[9000, :]}')
            # print(f'Y: {Y[9000, :]}')
            self.model.fit(X_cat, Y)

    def get_bottleneck_fct(self):
        # using the first d_bottleneck entries...double check if this makes sense
        # TODO: this should be the d_bottleneck lin indep rows!
        try:
            # print(self.model.coef_)
            linear_map = self.model.coef_[:self.d_bottleneck, :self.X_dim].T
        except AttributeError:  # go here if self.X_dim is not defined, i.e. no conditioning
            linear_map = self.model.coef_[:self.d_bottleneck, :].T
        print(f'Linear map:\n{linear_map}')
        fct = lambda x: x @ linear_map

        return fct


def get_children(A, i):
    """
    Args:
        A: np.array
            Adjacency matrix.
        i: int
            Index of node to get children of.

    Returns:
        children: list[]
            List of indices of children of node at index i.
    """
    return np.nonzero(A[i,:])[0]


def sort_parent_idxs(parent_idxs, causal_order):
    order_idxs = [np.where(causal_order == i)[0][0] for i in parent_idxs]
    return [i for _,i in sorted(zip(order_idxs, parent_idxs))]


def _get_var_idx(SCBM, var):
    return np.where(SCBM.variables == var)[0][0]


def check_open_path(source, target):
    """
    Check if there is an open path between the start and
    end nodes. Probably this is pretty inefficient.
    """
    if source in target.parents:
        return True

    # Check that candidate node has parents
    if target.parents is not None:
        # Only consider candidates that themselves have parents
        candidates = [i for i in target.parents if i.parents is not None]
        for candidate in candidates:
            intermed = check_open_path(source, candidate)
            if intermed:
                return True
    else:
        return False

    return False


def get_cond_set(source, target, SCBM, causal_order, bottlenecks):
    # print('Cond. set:')
    backdoor_cond_set = []
    frontdoor_cond_set = []

    source_idx = _get_var_idx(SCBM, source)
    target_idx = _get_var_idx(SCBM, target)
    # Get backdoor conditioning set
    if source.parents is None:
        pass
    else:
        for parent in source.parents:
            parent_idx = _get_var_idx(SCBM, parent)
            # print(f'backdoor: {parent_idx}')
            # print(f'getting bottleneck [{parent_idx}, {source_idx}]')
            bottleneck_fct = bottlenecks[parent_idx, source_idx]
            backdoor_cond_set.append(bottleneck_fct(parent.value))

    # Get frontdoor conditioning set
    target_parent_idxs = [_get_var_idx(SCBM, var) for var in target.parents]
    target_parent_idxs_sorted = sort_parent_idxs(target_parent_idxs, causal_order)
    # Only take parents later in causal ordering than source node
    target_parent_idxs_sub = target_parent_idxs_sorted[(target_parent_idxs_sorted.index(source_idx) + 1):]
    # Do nothing if the list is now empty
    if not target_parent_idxs_sub:
        pass
    else:
        for target_parent_idx in target_parent_idxs_sub:
            target_parent = SCBM.variables[target_parent_idx]
            bottleneck_fct = bottlenecks[target_parent_idx, target_idx]
            frontdoor_cond_set.append(bottleneck_fct(target_parent.value))

    cond_set = backdoor_cond_set + frontdoor_cond_set

    if not cond_set:  # return empty set
        # print('None')
        return cond_set
    else:
        return np.concatenate(cond_set, axis=1)


def estimate_bottleneck_fcts(SCBM, mode='linear'):
    # Matrix to save the estimated bottleneck functions.
    # Save these according to adjacency matrix.
    estimated_bottleneck_fcts = np.empty_like(SCBM.A, dtype=object)

    # Get causal order from adjacency matrix
    # TODO: implement this! Assuming upper triangular A matrix for now.
    causal_order = np.arange(SCBM.A.shape[0])

    # Choose corresponding regressor depending on mode
    # mode = 'linear'
    if mode == 'linear':
        reg_model = LinRegressor
    else:
        pass

    # Loop over variables (in special ordering) and estimate bottleneck function
    # Outer loop over target nodes
    for target_idx in causal_order:
        # print(f'source: {i}')
        target = SCBM.variables[target_idx]

        # If target is root node, skip (doesn't have parent bottlenecks)
        if target.parents is None:
            pass
        else:
            # Loop over parents in reverse causal order
            parent_idxs = [_get_var_idx(SCBM, parent) for parent in target.parents]
            parent_idxs_sort = sort_parent_idxs(parent_idxs, causal_order)
            for source_idx in reversed(parent_idxs_sort):
                # print(f'target: {child_idx}')
                source = SCBM.variables[source_idx]
                cond_set = get_cond_set(source, target, SCBM, causal_order,
                                        estimated_bottleneck_fcts)
                d_cond = cond_set.shape[1] if len(cond_set) > 0 else 0
                # print(f'd_cond: {d_cond}')

                regressor = reg_model(d_micro_in=source.d + d_cond,
                                      d_micro_out=target.d,
                                      d_bottleneck=SCBM.d_bottleneck_matrix[source_idx, target_idx])
                regressor.fit(X=source.value, Y=target.value, X_cond=cond_set)
                bottleneck_fct = regressor.get_bottleneck_fct()
                estimated_bottleneck_fcts[source_idx, target_idx] = bottleneck_fct

    return estimated_bottleneck_fcts


def main():
    # seed = 0
    # rs = np.random.RandomState(seed=seed)
    #
    # d_micro = 4
    # d_bottleneck = 1
    #
    # A, test_scbm = def_scbm(rs, d_micro, seed)
    # test_scbm.sample(size=50000)
    sampler = SCBMSampler(seed=0,
                          d_macro=4,
                          d_micro=4,
                          d_bottleneck=1,
                          bottleneck_mode='convex_comb',
                          mech_mode='constant',
                          p=0.8)

    test_scbm = sampler.sample()

    obs_sample = test_scbm.sample(size=50000)

    estimated_bottlenecks = estimate_bottleneck_fcts(test_scbm)

    a = 0


if __name__ == '__main__':
    main()