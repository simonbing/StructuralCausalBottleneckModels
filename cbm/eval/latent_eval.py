import numpy as np
from sklearn.metrics import r2_score


def linear_bottleneck_eval(estimated_bottleneck_samples, gt_bottleneck_samples):
    assert estimated_bottleneck_samples.shape == gt_bottleneck_samples.shape, \
        'Estimated bottleneck functions and g.t. must have same shape!'

    d = estimated_bottleneck_samples.shape[0]

    r2_matrix = np.empty_like(estimated_bottleneck_samples, dtype=object)

    for i in range(d):
        for j in range(d):
            if estimated_bottleneck_samples[i, j] is not None:
                # Get linear fit
                r2 = r2_score(gt_bottleneck_samples[i, j], estimated_bottleneck_samples[i, j])
                r2_matrix[i, j] = r2

    return r2_matrix
