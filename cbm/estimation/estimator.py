import numpy as np
from sklearn.preprocessing import StandardScaler

from cbm.estimation.lin_regressors import LinRegressor, ReducedRankRegressor
from cbm.estimation.ae_regressor import AutoencoderRegressor, VariationalAutoencoderRegressor
from cbm.estimation.utils import _get_var_idx, sort_parent_idxs


def get_cond_set(source, target, SCBM, samples, causal_order, estimated_bn_fcts,
                 no_bottlenecks=False):
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
            if no_bottlenecks:
                backdoor_cond_set.append(samples[parent_idx])
            else:
                bottleneck_fct = estimated_bn_fcts[parent_idx, source_idx]
                backdoor_cond_set.append((bottleneck_fct(samples[parent_idx])))

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
            if no_bottlenecks:
                frontdoor_cond_set.append(samples[target_parent_idx])
            else:
                bottleneck_fct = estimated_bn_fcts[target_parent_idx, target_idx]
                frontdoor_cond_set.append(bottleneck_fct(samples[target_parent_idx]))

    cond_set = backdoor_cond_set + frontdoor_cond_set

    if not cond_set:  # return empty set
        return cond_set
    else:
        return np.concatenate(cond_set, axis=1)


def estimate_bottleneck_and_mechanism_fcts(SCBM, samples, mode='linear',
                                           assumed_d_bn=None):
    # Matrix to save the estimated bottleneck functions.
    # Save these according to adjacency matrix.
    estimated_bottleneck_fcts = np.empty_like(SCBM.A, dtype=object)
    estimated_mechanism_fcts = np.empty_like(SCBM.A, dtype=object)

    # Get causal order from adjacency matrix
    # TODO: implement this! Assuming upper triangular A matrix for now.
    causal_order = np.arange(SCBM.A.shape[0])

    # Choose corresponding regressor depending on mode
    # mode = 'linear'
    if mode == 'linear':
        reg_model = LinRegressor
    elif mode == 'reduced_rank':
        reg_model = ReducedRankRegressor
    elif mode == 'mlp':
        reg_model = AutoencoderRegressor
    elif mode == 'vae':
        reg_model = VariationalAutoencoderRegressor
    else:
        pass

    # Loop over variables (in special ordering) and estimate bottleneck function
    # Outer loop over target nodes
    for target_idx in causal_order:
        target = SCBM.variables[target_idx]

        # If target is root node, skip (doesn't have parent bottlenecks)
        if target.parents is None:
            pass
        else:
            # Loop over parents in reverse causal order
            parent_idxs = [_get_var_idx(SCBM, parent) for parent in target.parents]
            parent_idxs_sort = sort_parent_idxs(parent_idxs, causal_order)
            for source_idx in reversed(parent_idxs_sort):
                source = SCBM.variables[source_idx]
                cond_set = get_cond_set(source, target, SCBM, samples, causal_order,
                                        estimated_bottleneck_fcts)
                d_cond = cond_set.shape[1] if len(cond_set) > 0 else 0

                reg_args = {'seed': SCBM.seed,
                            'd_micro_in': source.d,
                            'd_micro_out': target.d,
                            'd_bottleneck': SCBM.d_bottleneck_matrix[source_idx, target_idx] if not assumed_d_bn else assumed_d_bn,
                            'source': source_idx,
                            'target': target_idx,
                            'd_cond': d_cond}
                if mode in ['mlp', 'vae']:
                    mlp_args = {
                                'dense_x_z': [256, 128, 64, 64, 32],
                                'dense_z_x': [32, 64, 64, 128, 256],
                                'epochs': 500,
                                'batch_size': 1024,
                                'learning_rate': 0.00001,
                                'momentum': 0.9}
                    if mode == 'vae':
                        mlp_args['beta'] = 0.1  # Weight of KL term
                    reg_args = reg_args | mlp_args

                regressor = reg_model(**reg_args)

                regressor.fit(X=samples[source_idx],
                              Y=samples[target_idx],
                              X_cond=cond_set)
                # TODO: add saving of mechanism functions as well
                bottleneck_fct, mechanism_fct = regressor.get_bottleneck_and_mechanism_fcts()
                estimated_bottleneck_fcts[source_idx, target_idx] = bottleneck_fct
                estimated_mechanism_fcts[source_idx, target_idx] = mechanism_fct            

    return estimated_bottleneck_fcts, estimated_mechanism_fcts


def estimate_effects_ols(SCBM, samples):
    """
    Estimate the overall (linear) effect between all macro nodes of an SCBM.
    """
    # Matrix to save the estimated effect functions.
    # Save these according to adjacency matrix.
    estimated_effect_fcts = np.empty_like(SCBM.A, dtype=object)

    causal_order = np.arange(SCBM.A.shape[0])

    # Loop over variables (in special ordering) and estimate effect function
    # Outer loop over target nodes
    for target_idx in causal_order:
        target = SCBM.variables[target_idx]

        # If target is root node, skip (doesn't have parent bottlenecks)
        if target.parents is None:
            pass
        else:
            # Loop over parents in reverse causal order
            parent_idxs = [_get_var_idx(SCBM, parent) for parent in
                           target.parents]
            parent_idxs_sort = sort_parent_idxs(parent_idxs, causal_order)
            for source_idx in reversed(parent_idxs_sort):
                # print(f'target: {child_idx}')
                source = SCBM.variables[source_idx]
                cond_set = get_cond_set(source, target, SCBM, samples, causal_order,
                                        None, no_bottlenecks=True)
                d_cond = cond_set.shape[1] if len(cond_set) > 0 else 0

                regressor = LinRegressor(seed=SCBM.seed,
                                         d_micro_in=source.d,
                                         d_micro_out=target.d,
                                         d_bottleneck=target.d,
                                         source=source_idx,
                                         target=target_idx,
                                         d_cond=d_cond)

                regressor.fit(X=samples[source_idx],
                              Y=samples[target_idx],
                              X_cond=cond_set)
                effect_fct, _ = regressor.get_bottleneck_and_mechanism_fcts()

                estimated_effect_fcts[source_idx, target_idx] = effect_fct

    return estimated_effect_fcts
