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
                # Get linear fit
                regr = LinearRegression()
                regr.fit(estimated_bottleneck_samples[i, j][:n_train, ...],
                         gt_bottleneck_samples[i, j][:n_train, ...] )
                score = regr.score(estimated_bottleneck_samples[i, j][n_train:, ...],
                                   gt_bottleneck_samples[i, j][n_train:, ...])
                r2_matrix[i, j] = score

    return r2_matrix


def nonlinear_bottleneck_eval(estimated_bottleneck_samples, gt_bottleneck_samples):
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
                # Fit regressor
                regr = MLPRegressor(seed=0, d=d_micro, dense_layers=[128, 128, 128, 128, 128, 128],
                                    learning_rate=0.005, momentum=0.9,
                                    epochs=100, batch_size=5000,
                                    source=i, target=j)
                # regr.fit(estimated_bottleneck_samples[i, j][:n_train, ...],
                #          gt_bottleneck_samples[i, j][:n_train, ...])
                # score = regr.score(estimated_bottleneck_samples[i, j][n_train:, ...],
                #                    gt_bottleneck_samples[i, j][n_train:, ...])
                # Sanity check 3: apply known bijection
                # lin_map = np.asarray([[1, 1], [1, -1]])
                # tf_sample = 0.1 * (gt_bottleneck_samples[i, j] @ lin_map) ** 3
                # rescale data
                scaler = StandardScaler()
                tf_sample = estimated_bottleneck_samples[i, j]
                tf_scale = scaler.fit_transform(tf_sample)
                gt_scale = scaler.fit_transform(gt_bottleneck_samples[i, j])
                # tf_sample = copy.deepcopy(gt_bottleneck_samples[i, j])
                # regr.fit(tf_sample[:n_train, ...],
                #          gt_bottleneck_samples[i, j][:n_train, ...])
                # score = regr.score(
                #     tf_sample[n_train:, ...],
                #     gt_bottleneck_samples[i, j][n_train:, ...])
                regr.fit(tf_scale[:n_train, ...],
                         gt_scale[:n_train, ...])
                score = regr.score(
                    tf_scale[n_train:, ...],
                    gt_scale[n_train:, ...])
                # Sanity check 2: random sample as input to regr
                # rand_sample = np.random.rand(*estimated_bottleneck_samples[i, j].shape)
                # rand_sample = np.zeros_like(estimated_bottleneck_samples[i, j])
                # regr.fit(rand_sample[:n_train, ...],
                #          gt_bottleneck_samples[i, j][:n_train, ...])
                # score = regr.score(rand_sample[n_train:, ...],
                #                    gt_bottleneck_samples[i, j][n_train:, ...])
                # Sanity check 1: random sample and compute score
                # rand_sample = np.random.rand(*estimated_bottleneck_samples[i, j][n_train:, ...].shape)
                # score = np.sum((rand_sample - gt_bottleneck_samples[i, j][n_train:, ...]) ** 2, axis=1).mean()
                mse_matrix[i, j] = score

    return mse_matrix
