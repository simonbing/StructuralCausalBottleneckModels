import numpy as np
from sklearn.preprocessing import StandardScaler

from cbm.estimation.lin_regressors import LinRegressor, ReducedRankRegressor
from cbm.estimation.ae_regressor import AutoencoderRegressor, VariationalAutoencoderRegressor
from cbm.estimation.utils import _get_var_idx, sort_parent_idxs


def get_cond_set(source, target, SCBM, samples, causal_order, estimated_bn_fcts,
                 no_bottlenecks=False):
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
            if no_bottlenecks:
                # backdoor_cond_set.append(parent.value)
                backdoor_cond_set.append(samples[parent_idx])
            else:
                bottleneck_fct = estimated_bn_fcts[parent_idx, source_idx]
                # backdoor_cond_set.append(bottleneck_fct(parent.value))
                backdoor_cond_set.append((bottleneck_fct(samples[parent_idx])))
                # DEBUG
                # backdoor_cond_set.append(SCBM.bottleneck_samples[parent_idx, source_idx])
                # backdoor_cond_set.append(parent.value)

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
                # frontdoor_cond_set.append(target_parent.value)
                frontdoor_cond_set.append(samples[target_parent_idx])
            else:
                bottleneck_fct = estimated_bn_fcts[target_parent_idx, target_idx]
                # frontdoor_cond_set.append(bottleneck_fct(target_parent.value))
                frontdoor_cond_set.append(bottleneck_fct(samples[target_parent_idx]))
                # DEBUG
                # frontdoor_cond_set.append(SCBM.bottleneck_samples[target_parent_idx, target_idx])
                # frontdoor_cond_set.append(target_parent.value)

    cond_set = backdoor_cond_set + frontdoor_cond_set

    if not cond_set:  # return empty set
        # print('None')
        return cond_set
    else:
        # Experimental
        # scaler = StandardScaler()
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
                cond_set = get_cond_set(source, target, SCBM, samples, causal_order,
                                        estimated_bottleneck_fcts)
                # cond_set = []
                d_cond = cond_set.shape[1] if len(cond_set) > 0 else 0
                # DEBUG
                # if len(cond_set) > 0:
                #     scaler = StandardScaler()
                #     cond_set = scaler.fit_transform(cond_set)
                # print(f'd_cond: {d_cond}')

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
                                # 'dense_x_z': [128, 128, 128, 128, 128, 128],
                                # 'dense_z_x': [128, 128, 128, 128, 128, 128],
                                'epochs': 500,
                                # 'batch_size': 512,
                                # 'learning_rate': 0.000005,
                                'batch_size': 1024,
                                'learning_rate': 0.00001,
                                # 'learning_rate': 0.00005,
                                'momentum': 0.9}
                    if mode == 'vae':
                        mlp_args['beta'] = 0.1  # Weight of KL term
                    reg_args = reg_args | mlp_args

                regressor = reg_model(**reg_args)

                ### DEBUG
                # rs = np.random.RandomState(0)
                # # Reproduce mechanism
                # from cbm.data.utils import rand_undirected_adj_matrix, \
                #     sample_mrf_prec
                #
                # M = rand_undirected_adj_matrix(rs=rs, nodes=4)
                # P = sample_mrf_prec(dim=4, M=M, rs=rs)
                # E = np.linalg.inv(P)
                # L = np.linalg.cholesky(E)
                # mu_i = rs.normal(size=source.value.shape)
                # X_i = (L @ mu_i.T).T
                #
                # # X_i = rs.normal(size=source.value.shape)
                # W_i = rs.uniform(size=(4, 2))
                # Z = (1 * (X_i @ W_i)) ** 3
                # W_j = rs.uniform(size=(2, 4))
                # X_j = (1 * (Z @ W_j)) ** 3
                #
                # M_j = rand_undirected_adj_matrix(rs=rs, nodes=4)
                # P_j = sample_mrf_prec(dim=4, M=M_j, rs=rs)
                # E_j = np.linalg.inv(P_j)
                # L_j = np.linalg.cholesky(E_j)
                # mu_j = rs.normal(size=source.value.shape)
                # noise_j = (L_j @ mu_j.T).T
                # X_j = X_j + noise_j
                #
                # from sklearn.preprocessing import StandardScaler
                # scaler = StandardScaler()
                #
                # X_i = scaler.fit_transform(X_i)
                # X_j = scaler.fit_transform(X_j)
                # # X_j = X_j + rs.normal(size=X_j.shape)
                #
                # regressor.fit(X=X_i, y=X_j, X_cond=cond_set)
                #########

                # regressor.fit(X=source.value, y=target.value, X_cond=cond_set)
                regressor.fit(X=samples[source_idx],
                              Y=samples[target_idx],
                              X_cond=cond_set)
                # TODO: add saving of mechanism functions as well
                bottleneck_fct, mechanism_fct = regressor.get_bottleneck_and_mechanism_fcts()
                estimated_bottleneck_fcts[source_idx, target_idx] = bottleneck_fct
                estimated_mechanism_fcts[source_idx, target_idx] = mechanism_fct

                ### DEBUG
                # Z_hat = bottleneck_fct(X_i)
                #
                # from cbm.eval.mlp_regressor import MLPRegressor
                #
                # regr_forward = MLPRegressor(seed=0,
                #                             d=2,
                #                             dense_layers=[128, 128, 128, 128,
                #                                           128, 128],
                #                             learning_rate=0.0005,
                #                             momentum=0.9,
                #                             epochs=100,
                #                             batch_size=5000,
                #                             source=0,
                #                             target=1)
                #
                # regr_back = MLPRegressor(seed=0,
                #                          d=2,
                #                          dense_layers=[128, 128, 128, 128,
                #                                        128, 128],
                #                          learning_rate=0.0005,
                #                          momentum=0.9,
                #                          epochs=100,
                #                          batch_size=5000,
                #                          source=0,
                #                          target=1)
                #
                # n_train = int(0.8 * len(Z_hat))
                #
                # # Z_hat = Z ** 3
                #
                # Z = scaler.fit_transform(Z)
                # Z_hat = scaler.fit_transform(Z_hat)
                #
                # import matplotlib.colors
                # import matplotlib.pyplot as plt
                # # Color map
                # def get_rgb_color(x, y):
                #     hue = (np.arctan2(y, x) + np.pi) / (2*np.pi)
                #     saturation = np.sqrt(x ** 2 + y ** 2)
                #     # Standardize
                #     saturation = saturation / np.max(saturation)
                #     value = np.ones_like(hue)
                #
                #     colors = matplotlib.colors.hsv_to_rgb(np.stack((hue, saturation, value)).T)
                #
                #     return colors
                #
                # # colors = get_rgb_color(Z[:, 0], Z[:, 1])
                # #
                # # plt.scatter(Z[:, 0], Z[:, 1], c=colors)
                # # plt.show()
                #
                # regr_forward.fit(Z_hat[:n_train, ...], Z[:n_train, ...])
                # score_forward = regr_forward.score(Z_hat[n_train:, ...], Z[n_train:, ...])
                #
                # regr_back.fit(Z[:n_train, ...], Z_hat[:n_train, ...])
                # score_back = regr_back.score(Z[n_train:, ...], Z_hat[n_train:, ...])
                # a=0
                #########

    return estimated_bottleneck_fcts, estimated_mechanism_fcts


def estimate_effects_ols(SCBM, samples):
    """
    Estimate the overall (linear) effect between all macro nodes of an SCBM.
    :param SCBM:
    :return:
    """
    # Matrix to save the estimated effect functions.
    # Save these according to adjacency matrix.
    estimated_effect_fcts = np.empty_like(SCBM.A, dtype=object)

    causal_order = np.arange(SCBM.A.shape[0])

    # Loop over variables (in special ordering) and estimate effect function
    # Outer loop over target nodes
    for target_idx in causal_order:
        # print(f'source: {i}')
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

                # regressor.fit(X=source.value, y=target.value, X_cond=cond_set)
                regressor.fit(X=samples[source_idx],
                              Y=samples[target_idx],
                              X_cond=cond_set)
                effect_fct, _ = regressor.get_bottleneck_and_mechanism_fcts()

                estimated_effect_fcts[source_idx, target_idx] = effect_fct

    return estimated_effect_fcts
