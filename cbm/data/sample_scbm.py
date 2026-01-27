from collections.abc import Iterable

import numpy as np

from cbm.data.utils import rand_weight_matrix, rand_undirected_adj_matrix, sample_mrf_prec
from cbm.data.bottlenecks import sample_convex_comb_bottleneck, \
    sample_lin_bottleneck, manual_nonlinear, sample_nonlin_bottleneck
from cbm.data.mechanisms import constant_scalar_mechanism, linear_mechanism, \
    manual_nonlinear_mechanism, sample_nonlin_mechanism
from cbm import SCBM, GaussianLangevinMechanism, MacroCausalVar


class SCBMSampler(object):
    def __init__(self, seed, d_macro, d_micro, d_bottleneck, bottleneck_mode,
                 mech_mode, p=0.3):
        """
        TODO: add docstring
        """
        self.seed = seed
        self.rs = np.random.RandomState(seed=self.seed)

        self.d_macro = d_macro

        # Sample adjacency matrix
        self.A = np.ceil(rand_weight_matrix(seed=self.seed,
                                            nodes=self.d_macro,
                                            connect_prob=p))

        # Repeat values if not a list
        if not isinstance(d_micro, Iterable):
            self.d_micro = d_micro * np.ones(d_macro, dtype=int)
        else:
            self.d_micro = d_micro
        if not isinstance(d_bottleneck, Iterable):
            n_edges = np.count_nonzero(self.A)
            self.d_bottleneck = d_bottleneck * np.ones(n_edges, dtype=int)
        else:
            self.d_bottleneck = d_bottleneck

        self.d_bottleneck_matrix = np.empty_like(self.A, dtype=object)
        counter = 0
        for i in range(self.d_macro):
            for j in range(self.d_macro):
                if self.A[i, j] == 1:
                    self.d_bottleneck_matrix[i, j] = self.d_bottleneck[counter]
                    counter += 1

        self.causal_order = np.arange(self.d_macro)

        self.bottleneck_sampler = self._get_bottleneck_sampler(bottleneck_mode)

        self.mechanism_sampler = self._get_mechanism_sampler(mech_mode)

    def _get_bottleneck_sampler(self, mode):
        if mode == 'convex_comb':
            return sample_convex_comb_bottleneck
        elif mode == 'linear':
            return sample_lin_bottleneck
        elif mode == 'nonlinear':
            return sample_nonlin_bottleneck
        elif mode == 'manual_nonlinear':
            return manual_nonlinear
        else:
            ValueError, f'bottleneck_mode {mode} not defined!'

    def _get_mechanism_sampler(self, mode):
        if mode == 'constant':
            return constant_scalar_mechanism
        elif mode == 'linear':
            return linear_mechanism
        elif mode == 'nonlinear':
            return sample_nonlin_mechanism
        elif mode == 'manual_nonlinear':
            return manual_nonlinear_mechanism
        else:
            ValueError, f'mech_mode {mode} not defined!'

    def _get_bottleneck_fcts(self):
        b_fcts = np.empty_like(self.A, dtype=object)

        for i in range(self.d_macro):
            for j in range(self.d_macro):
                if self.A[i, j] == 1:
                    b_fcts[i, j] = self.bottleneck_sampler(
                        rs=self.rs,
                        d_micro=self.d_micro[i],
                        d_bottleneck=self.d_bottleneck_matrix[i, j])

        return b_fcts

    def _get_mechanism_fcts(self):
        m_fcts = np.empty(self.d_macro, dtype=object)

        for i in range(self.d_macro):
            # Need to get the bottleneck dimensions of all parents
            # Get parents
            parent_idxs = np.nonzero(self.A[:, i])[0]
            if parent_idxs.size == 0:
                d_bottleneck = None
            else:
                d_bottleneck = self.d_bottleneck_matrix[parent_idxs, i]

            m_fcts[i] = self.mechanism_sampler(rs=self.rs,
                                               d_bottleneck=d_bottleneck,
                                               d_micro=self.d_micro[i])

        return m_fcts

    def sample(self):
        # Get bottleneck functions
        bottleneck_fcts = self._get_bottleneck_fcts()
        # Get mechanism functions
        mechanism_fcts = self._get_mechanism_fcts()

        # Define variables
        variables = np.empty(self.d_macro, dtype=object)
        for i in range(self.d_macro):
            # Sample internal adjacency matrix
            M = rand_undirected_adj_matrix(rs=self.rs, nodes=self.d_micro[i])

            # Sample precision matrix for internal mechanism
            P = sample_mrf_prec(dim=self.d_micro[i], M=M, rs=self.rs)

            # print(f"P{i}: {P}")

            parents = self.A[:, i]
            if np.sum(parents) == 0:  # root node
                mech = GaussianLangevinMechanism(mu=np.zeros(self.d_micro[i]),
                                                 E=np.linalg.inv(P))
                variables[i] = MacroCausalVar(parents=None,
                                              bottleneck_fcts=None,
                                              mechanism=mech,
                                              d=self.d_micro[i])
            else:
                parent_idxs = np.nonzero(parents)[0]
                mech = GaussianLangevinMechanism(mu=mechanism_fcts[i],
                                                 E=np.linalg.inv(P))
                variables[i] = MacroCausalVar(parents=variables[parent_idxs],
                                              bottleneck_fcts=[
                                                  bottleneck_fcts[p_idx, i] for
                                                  p_idx in parent_idxs],
                                              mechanism=mech,
                                              d=self.d_micro[i])

        scbm = SCBM(variables=variables, A=self.A,
                    d_bottleneck_matrix=self.d_bottleneck_matrix,seed=self.seed)

        return scbm
