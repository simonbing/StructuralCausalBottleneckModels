import numpy as np

from cbm.data.utils import rand_weight_matrix, rand_undirected_adj_matrix, sample_mrf_prec
from cbm import SCBM, GaussianLangevinMechanism, MacroCausalVar


# Transfer learning experiment
def get_SCBM_tf_1(seed, d):
    rs = np.random.RandomState(seed
                               )
    # Sample internal adjacency matrix
    M_C = rand_undirected_adj_matrix(rs=rs, nodes=d)
    # Sample precision matrix for internal mechanism
    P_C = sample_mrf_prec(dim=d, M=M_C, rs=rs)

    mech_C = GaussianLangevinMechanism(mu=np.zeros(d),
                                       E=np.linalg.inv(P_C))
    C = MacroCausalVar(parents=None,
                       bottleneck_fcts=None,
                       mechanism=mech_C,
                       d=d)

    M_X = rand_undirected_adj_matrix(rs=rs, nodes=1)
    P_X = sample_mrf_prec(dim=1, M=M_X, rs=rs)

    alpha = 1.0

    mech_X = GaussianLangevinMechanism(mu=lambda x: alpha * x,
                                       E=np.linalg.inv(P_X))

    # Sample from simplex
    k_X = rs.exponential(scale=1.0, size=[d, 1])
    W_X = k_X / sum(k_X)

    def bottleneck_fct_X(C_in):
        return C_in @ W_X

    X = MacroCausalVar(parents=[C],
                       bottleneck_fcts=[bottleneck_fct_X],
                       mechanism=mech_X,
                       d=1)

    # Sample from simplex
    k_Y = rs.exponential(scale=1.0, size=[d, 1])
    W_Y = k_Y / sum(k_Y)

    def bottleneck_fct_Y(C_in):
        return C_in @ W_Y

    M_Y = rand_undirected_adj_matrix(rs=rs, nodes=1)
    P_Y = sample_mrf_prec(dim=1, M=M_Y, rs=rs)

    delta = 1.0
    gamma = 1.0

    mech_Y = GaussianLangevinMechanism(mu=lambda x1, x2: delta * x1 + gamma * x2,
                                       E=np.linalg.inv(P_Y))

    Y = MacroCausalVar(parents=[C, X],
                       bottleneck_fcts=[bottleneck_fct_Y, lambda x: x],
                       mechanism=mech_Y,
                       d=1)

    variables = np.array([C, X, Y])

    A = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]])

    d_bottleneck_matrix = np.array([[None, 1, 1],
                                    [None, None, 1],
                                    [None, None, None]])

    out_scbm = SCBM(variables=variables,
                    A=A,
                    d_bottleneck_matrix=d_bottleneck_matrix,
                    seed=seed)

    return out_scbm
