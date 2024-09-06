import numpy as np


def _get_var_idx(SCBM, var):
    return np.where(SCBM.variables == var)[0][0]


def sort_parent_idxs(parent_idxs, causal_order):
    order_idxs = [np.where(causal_order == i)[0][0] for i in parent_idxs]
    return [i for _,i in sorted(zip(order_idxs, parent_idxs))]
