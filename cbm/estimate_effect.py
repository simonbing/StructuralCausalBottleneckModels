import numpy as np

from cbm.estimation.lin_regressors import LinRegressor, ReducedRankRegressor
from cbm.estimation.estimator import get_cond_set


def define_linear_effect(bn_fct, mech_fct):

    def effect(x):
        return mech_fct(bn_fct(x))
    return effect


def estimate_effect(SCBM, source_idx, target_idx, estimated_bn_fcts, estimated_mech_fcts,
                    mode, reestimate_effect=False, verbose=False):
    source_var = SCBM.variables[source_idx]
    target_var = SCBM.variables[target_idx]

    if mode == 'linear':
        if not reestimate_effect:
            # Direct effect from source to target should always be included
            indiv_effects = [define_linear_effect(estimated_bn_fcts[source_idx, target_idx],
                                                  estimated_mech_fcts[source_idx, target_idx])]
            # Get indices of variables in the conditioning set
            cond_idxs = None
            for cond_idx in cond_idxs:
                pass

            def total_effect_fct(*args):
                out_list = []
                for effect, arg in zip(indiv_effects, args):
                    out_list.append(effect(arg))

                return np.sum(out_list)



    pass