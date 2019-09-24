import numpy as np
import scipy.sparse as sp

# weights
def init_Weight(shape):
    """Glorot & Bengio (AISTATS 2010) init"""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return initial


def get_Weight_from_file(weight_str):
    if weight_str == "weights_outputs":
        m, n = 16, 7
    elif weight_str == "weights_hidden":
        m, n = 1433, 16
    elif weight_str == "hidden":
        m, n = 2708, 16
    else:
        m, n = 2708, 7
    ans = np.zeros((m, n), dtype=np.float32)
    weight_path = "../weights/{}.txt".format(weight_str)
    with open(weight_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            for j, v in enumerate(line.strip().split(" ")):
                ans[i, j] = float(v)
    return ans


def init_dropout(shape, dropout):
    """Dropout 2014, * input"""
    col = np.array([1] * shape[0]).reshape(-1, 1)
    mat = np.repeat(col, shape[1], axis=1)
    return np.random.binomial(mat, dropout)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

# prepare features
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

# prepare adj
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()
