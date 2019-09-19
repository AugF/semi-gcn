import numpy as np
import pickle as pkl
import sys
import scipy.sparse as sp
import networkx as nx


def onehot(y, c):
    res = np.zeros((len(y), c))
    for i, j in enumerate(y):
        res[i, j] = 1
    return res


def l2_loss(X):
    """for matrix, tf.nn.l2_loss: np.sum(x**2)/2"""
    x_square = X ** 2
    x_sum = np.sum(x_square)
    x_l2 = x_sum / 2
    return x_l2


def parse_index_file(filename):
    # ind.dataset.test.index
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    # idx: sample_list;   l: total length
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def prepare_gcn():
    # return A, P, X, W0, W1, Y, train_mask
    # just for train
    n, f, h, c =10, 4, 2, 3

    np.random.seed(1)
    A = np.random.random((n, n))  #
    P = np.random.random((n, n))  # reg
    X = np.random.random((n, f)) # features

    W0 = np.random.random((f, h))
    W1 = np.random.random((h, c))

    y = np.random.randint(0, c, (n, ))
    Y = onehot(y, c)

    # prepare y_mask
    l = 2
    train_mask = sample_mask(range(l), n)

    return A, P, X, W0, W1, Y, train_mask


def numerical_grad(f, X, h=1e-5):
    grad = np.zeros(X.shape)
    m, n = X.shape

    for i in range(m):
        for j in range(n):
            X[i, j] += h
            loss1 = f(X)
            X[i, j] -= 2*h
            loss2 = f(X)
            grad[i, j] = (loss1 - loss2) / (2*h)
            X[i, j] += h
    return grad

def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    # cora
    # x: csr_matrix  (140, 1433)  y: array  (140, 7)  train data: labeled data
    # tx: csr_matrix (1000, 1433) , ty    test data: labeled data
    # allx: csr (1708, 1433)  ally
    # labeled data(1640):  [0, 140] train_data  [140, 140+500] val_data (fix at 500)  [2708-1000, 2708] test_data
    # unlabeled data(1068):  [740, 1708]
    # g: defaultdict (2708， ?)

    # y is one-hot (n, 7)
    # 1. get adj
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # csr_matrix,  (2708, 2708) row compressed toarray

    index = parse_index_file("../data/ind.{}.test.index".format(dataset_str))  # 1708~2707 乱序

    # 2. get ordered features and labes
    features = sp.vstack((allx, tx)).tolil()  # lil_matrix, todense  position
    index_sorted = np.sort(index)
    features[index, ] = features[index_sorted, ]

    labels = np.vstack((ally, ty))
    labels[index, ] = labels[index_sorted, ]

    # 3. get mask (n, 1)
    train_list = np.arange(0, len(y))
    val_list = np.arange(len(y), len(y) + 500)
    test_list = np.arange(len(ally), len(ally) + len(ty))

    train_mask = sample_mask(train_list, labels.shape[0])
    val_mask = sample_mask(val_list, labels.shape[0])
    test_mask = sample_mask(test_list, labels.shape[0])

    # y_val[train_mask, ] = labels[train_mask, ]  mask的使用方法
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask





