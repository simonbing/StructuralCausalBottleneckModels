import numpy as np


def leaky_relu(x):
    if x < 0:
        return 0.1 * x
    else:
        return x
    
def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample_mlp(rs, in_dim, out_dim, hidden_dim, hidden_layers,
               nonlinearity='leaky_relu'):
    if nonlinearity == 'leaky_relu':
        nonlin_f = np.vectorize(leaky_relu)
    elif nonlinearity == 'relu':
        nonlin_f = np.vectorize(relu)
    elif nonlinearity == 'sigmoid':
        nonlin_f = np.vectorize(sigmoid)
    elif nonlinearity == 'swish':
        nonlin_f = np.vectorize(lambda x: x * sigmoid(x))
    elif nonlinearity == 'none':
        nonlin_f = np.vectorize(lambda x: x)
    else:
        raise ValueError

    w_list = []
    for i in range(hidden_layers):
        if i == 0:
            w = rs.uniform(size=(in_dim, hidden_dim))
            # w = rs.uniform(-1, 1, size=(in_dim, hidden_dim))
            # w = rs.normal(loc=1.0, scale=1.0, size=(in_dim, hidden_dim))
            while np.linalg.matrix_rank(w) < min(in_dim, hidden_dim):
                w = rs.uniform(size=(in_dim, hidden_dim))
                # w = rs.uniform(-1, 1, size=(in_dim, hidden_dim))
                # w = rs.normal(loc=1.0, scale=1.0, size=(in_dim, hidden_dim))
        else:
            w = rs.uniform(size=(hidden_dim, hidden_dim))
            # w = rs.uniform(-1, 1, size=(hidden_dim, hidden_dim))
            # w = rs.normal(loc=1.0, scale=1.0, size=(hidden_dim, hidden_dim))
            while np.linalg.matrix_rank(w) < hidden_dim:
                w = rs.uniform(size=(hidden_dim, hidden_dim))
                # w = rs.uniform(-1, 1, size=(hidden_dim, hidden_dim))
                # w = rs.normal(loc=1.0, scale=1.0, size=(hidden_dim, hidden_dim))

        # QR decomposition to get better behaved weights
        Q, R = np.linalg.qr(w)
        w = Q

        w_list.append(w)

    w_out = rs.uniform(size=(hidden_dim, out_dim))
    # w_out = rs.uniform(-1, 1, size=(hidden_dim, out_dim))
    # w_out = rs.normal(loc=1.0, scale=1.0, size=(hidden_dim, out_dim))
    while np.linalg.matrix_rank(w_out) < min(hidden_dim, out_dim):
        w_out = rs.uniform(size=(hidden_dim, out_dim))
        # w_out = rs.uniform(-1, 1, size=(hidden_dim, out_dim))
        # w_out = rs.normal(loc=1.0, scale=1.0, size=(hidden_dim, out_dim))
    # QR decomposition to get better behaved weights
    if out_dim < hidden_dim:  # Only do QR for bottleneck mlp, otherwise dimension is too small
        Q, R = np.linalg.qr(w_out)
        w_out = Q

    def f(x):
        for i in range(hidden_layers):
            x = x @ w_list[i]
            x = nonlin_f(x)
        return x @ w_out

    return f


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
    # Sum of outer products of vectors make psd matrix
    P_list = []
    # Outer loop over rows
    for i in range(dim):
        # Inner loop over columns above diagonal
        for j in range(i+1, dim):
            if M[i, j] == 1:
                # Create sparsity mask
                m = np.zeros(dim)
                m[i] = m[j] = 1
                # Vector for outer product
                p = np.asarray([rs.random() if elem == 1 else 0.0 for elem in m])

                P_list.append(np.outer(p, p))

    P = np.sum(P_list, axis=0)
    eps = 0.01
    P += (eps * np.identity(dim))

    return P
