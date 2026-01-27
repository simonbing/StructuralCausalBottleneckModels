import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from cbm.eval.mlp_regressor import MLPRegressor

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
                n_train = int(train_frac * len(estimated_bottleneck_samples[i, j]))

                # Fit regressor (in both directions!)
                # Forward direction
                regr_forward = MLPRegressor(seed=0,
                                            d=estimated_bottleneck_samples[i, j].shape[1],
                                            d_out=gt_bottleneck_samples[i, j].shape[1],
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
                                         d=gt_bottleneck_samples[i, j].shape[1],
                                         d_out=estimated_bottleneck_samples[i, j].shape[1],
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
