import numpy as np

from cbm.estimation.lin_regressors import LinRegressor, ReducedRankRegressor
from cbm.estimation.mlp_regressor import MLPRegressor
from cbm.estimation.utils import _get_var_idx, sort_parent_idxs


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
    elif mode == 'reduced_rank':
        reg_model = ReducedRankRegressor
    elif mode == 'mlp':
        reg_model = MLPRegressor
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

                reg_args = {'seed': SCBM.seed,
                            'd_micro_in': source.d,
                            'd_micro_out': target.d,
                            'd_bottleneck': SCBM.d_bottleneck_matrix[source_idx, target_idx],
                            'd_cond': d_cond}
                # TODO: figure out a way to pass these args when calling the estimation function
                if mode == 'mlp':
                    mlp_args = {'dense_x_z': [64, 64],
                                'dense_z_x': [64, 64],
                                'epochs': 1,
                                'batch_size': 128,
                                'learning_rate': 0.005,
                                'momentum': 0.9}
                    reg_args = reg_args | mlp_args

                regressor = reg_model(**reg_args)
                regressor.fit(X=source.value, Y=target.value, X_cond=cond_set)
                bottleneck_fct = regressor.get_bottleneck_fct()
                estimated_bottleneck_fcts[source_idx, target_idx] = bottleneck_fct

    return estimated_bottleneck_fcts
