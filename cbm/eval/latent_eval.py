import copy

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from cbm.eval.mlp_regressor import MLPRegressor


def _compute_r2(z_gt, z_hat):
    z_hat = z_hat - np.mean(z_hat, axis=0, keepdims=True)
    z_gt = z_gt - np.mean(z_gt, axis=0, keepdims=True)
    scales = np.sum(z_gt * z_hat, axis=0, keepdims=True) / np.sum(
        z_hat * z_hat, axis=0, keepdims=True)
    return 1 - np.mean((z_gt - z_hat * scales) ** 2, axis=0) / np.mean(
        z_gt ** 2, axis=0)


def linear_bottleneck_eval(estimated_bottleneck_samples, gt_bottleneck_samples):
    assert estimated_bottleneck_samples.shape == gt_bottleneck_samples.shape, \
        'Estimated bottleneck functions and g.t. must have same shape!'

    d = estimated_bottleneck_samples.shape[0]

    r2_matrix = np.empty_like(estimated_bottleneck_samples, dtype=object)

    train_frac = 0.8

    for i in range(d):
        for j in range(d):
            if estimated_bottleneck_samples[i, j] is not None:
                n_train = int(train_frac * len(estimated_bottleneck_samples[i, j]))
                # Get linear fit (in both directions!)
                regr_forward = LinearRegression()
                regr_forward.fit(estimated_bottleneck_samples[i, j][:n_train, ...],
                                 gt_bottleneck_samples[i, j][:n_train, ...] )
                score_forward = regr_forward.score(estimated_bottleneck_samples[i, j][n_train:, ...],
                                                   gt_bottleneck_samples[i, j][n_train:, ...])

                regr_back = LinearRegression()
                regr_back.fit(gt_bottleneck_samples[i, j][:n_train, ...],
                              estimated_bottleneck_samples[i, j][:n_train, ...])
                score_back = regr_back.score(gt_bottleneck_samples[i, j][n_train:, ...],
                                             estimated_bottleneck_samples[i, j][n_train:, ...])

                # DEBUG
                pred_forward = regr_forward.predict(estimated_bottleneck_samples[i, j][n_train:, ...])
                mse_forward = np.mean((pred_forward - gt_bottleneck_samples[i, j][n_train:, ...]) ** 2)

                pred_back = regr_back.predict(gt_bottleneck_samples[i, j][n_train:, ...])
                mse_back = np.mean((pred_back - estimated_bottleneck_samples[i, j][n_train:, ...]) ** 2)

                # r2_matrix[i, j] = (mse_forward + mse_back) / 2

                r2_matrix[i, j] = (score_forward + score_back) / 2

    return r2_matrix


def nonlinear_bottleneck_eval(estimated_bottleneck_samples, gt_bottleneck_samples,
                              metric='r2'):
    assert estimated_bottleneck_samples.shape == gt_bottleneck_samples.shape, \
        'Estimated bottleneck functions and g.t. must have same shape!'

    d = estimated_bottleneck_samples.shape[0]

    mse_matrix = np.empty_like(estimated_bottleneck_samples, dtype=object)

    train_frac = 0.8

    for i in range(d):
        for j in range(d):
            if estimated_bottleneck_samples[i, j] is not None:
                d_micro = estimated_bottleneck_samples[i, j].shape[1]
                n_train = int(train_frac * len(estimated_bottleneck_samples[i, j]))

                # Sanity check 1: random sample and compute score
                # rand_sample = np.random.rand(*estimated_bottleneck_samples[i, j][n_train:, ...].shape)
                # score = np.sum((rand_sample - gt_bottleneck_samples[i, j][n_train:, ...]) ** 2, axis=1).mean()

                # Sanity check 2: random sample as input to regr
                # rand_sample = np.random.rand(*estimated_bottleneck_samples[i, j].shape)
                # rand_sample = np.zeros_like(estimated_bottleneck_samples[i, j])
                # source_forward = rand_sample
                # target_forward = gt_bottleneck_samples[i, j]
                #
                # source_back = gt_bottleneck_samples[i, j]
                # target_back = rand_sample

                # Sanity check 3: apply known bijection
                # lin_map = np.asarray([[1, 1], [1, -1]])
                # tf_sample = (gt_bottleneck_samples[i, j] @ lin_map) ** 3

                # Sanity check 4: apply injective transformation
                # lin_map = np.asarray([[1, 1], [1, -1]])
                # tf_sample = (gt_bottleneck_samples[i, j] @ lin_map) ** 2
                #
                # source_forward = tf_sample
                # target_forward = gt_bottleneck_samples[i, j]
                #
                # source_back = gt_bottleneck_samples[i, j]
                # target_back = tf_sample

                # Fit regressor (in both directions!)
                # Forward direction
                regr_forward = MLPRegressor(seed=0,
                                            d=d_micro,
                                            dense_layers=[128, 128, 128, 128, 128, 128],
                                            learning_rate=0.0005,
                                            momentum=0.9,
                                            epochs=100,
                                            batch_size=5000,
                                            source=i,
                                            target=j)

                source_forward = estimated_bottleneck_samples[i, j]
                target_forward = gt_bottleneck_samples[i, j]

                # Rescale data
                scaler_forward = StandardScaler()
                source_scale = scaler_forward.fit_transform(source_forward)
                target_scale = scaler_forward.fit_transform(target_forward)

                regr_forward.fit(source_scale[:n_train, ...],
                         target_scale[:n_train, ...])
                score_forward = regr_forward.score(
                    source_scale[n_train:, ...],
                    target_scale[n_train:, ...],
                    metric=metric)

                # Backward direction
                regr_back = MLPRegressor(seed=0,
                                         d=d_micro,
                                         dense_layers=[128, 128, 128, 128, 128, 128],
                                         learning_rate=0.0005,
                                         momentum=0.9,
                                         epochs=100,
                                         batch_size=5000,
                                         source=i,
                                         target=j)

                source_back = gt_bottleneck_samples[i, j]
                target_back = estimated_bottleneck_samples[i, j]

                # Rescale data
                scaler_back = StandardScaler()
                source_scale = scaler_back.fit_transform(source_back)
                target_scale = scaler_back.fit_transform(target_back)

                regr_back.fit(source_scale[:n_train, ...],
                                 target_scale[:n_train, ...])
                score_back = regr_back.score(
                    source_scale[n_train:, ...],
                    target_scale[n_train:, ...],
                    metric=metric)

                mse_matrix[i, j] = (score_forward + score_back) / 2

    return mse_matrix
