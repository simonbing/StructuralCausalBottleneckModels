import numpy as np
from sklearn.linear_model import LinearRegression


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
    n_train = int(train_frac * len(estimated_bottleneck_samples))

    for i in range(d):
        for j in range(d):
            if estimated_bottleneck_samples[i, j] is not None:
                # Get linear fit
                regr = LinearRegression()
                regr.fit(estimated_bottleneck_samples[i, j][:n_train, ...],
                         gt_bottleneck_samples[i, j][:n_train, ...])
                score = regr.score(estimated_bottleneck_samples[i, j][n_train:, ...],
                                   gt_bottleneck_samples[i, j][n_train:, ...])
                r2 = _compute_r2(gt_bottleneck_samples[i, j], estimated_bottleneck_samples[i, j])
                r2_matrix[i, j] = score
                # pr = pearsonr(gt_bottleneck_samples[i, j], estimated_bottleneck_samples[i, j])
                # a=0

    return r2_matrix
