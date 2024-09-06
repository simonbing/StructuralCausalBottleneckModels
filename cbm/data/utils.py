import numpy as np


def rand_weight_matrix(seed, nodes=3, connect_prob=0.5, wmin=0.1, wmax=1.0):
    """
    :param nodes: number of nodes
    :param connect_prob: probability of an edge
    :return: Upper diagonal weight matrix
    """
    rng = np.random.RandomState(seed=seed)

    adjacency_matrix = np.zeros([nodes, nodes], dtype=np.int32)  # [parents, nodes]
    weight_matrix = np.zeros([nodes, nodes], dtype=np.float32)  # [parents, nodes]

    causal_order = np.flip(np.arange(nodes))

    for i in range(nodes - 1):
        node = causal_order[i]
        potential_parents = causal_order[(i + 1):]
        num_parents = rng.binomial(n=nodes - i - 1, p=connect_prob)
        parents = rng.choice(potential_parents, size=num_parents,
                                   replace=False)
        adjacency_matrix[parents, node] = 1

    for i in range(nodes):
        for j in range(nodes):
            if adjacency_matrix[i, j] == 1:
                weight_matrix[i, j] = rng.uniform(wmin, wmax)

    return weight_matrix


def rand_undirected_adj_matrix(rs, nodes):
    """
    Args:
        rs: RandomState
        nodes: int
            Number of nodes.

    Returns:
        M: np.array
            Symmetric adjacancy matrix of an undirected graph.
    """
    U = rs.uniform(low=0, high=1.0, size=(nodes, nodes))
    S = np.tril(U) + np.tril(U, -1).T

    M = np.where(S > 0.5, 1, 0)
    np.fill_diagonal(M, 1)

    return M


def sample_mrf_prec(dim, M, rs):
    """
    Sample precision matrix of a Markov random field.
    Args:
        dim: int
            number of dimensions
        M: np.array
            array that encodes the sparsity structure of the precision matrix
            s.t. this is a valid MRF
        rs: RandomState
    Returns:
        P: np.array
            sampled precision matrix
    """
    def sample():
        P = rs.random(size=(dim, dim)) # sample dense dim x dim matrix
        P = P @ P.T # make this object symmetric
        P += np.identity(dim) # make positive definite
        P = np.where(M, P, 0) # apply sparsity map
        return P

    P = sample()
    # Rejection sampling until P is pd
    while any(np.linalg.eigvals(P) < 0):
        P = sample()

    return P