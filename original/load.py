import pickle as pkl
import scipy.sparse as sp
import numpy as np
import networkx as nx
import sys
from original.utils import onehot


def prepare_gcn():
    # adj, features, labels, train_mask
    n, f, c = 10, 4, 3

    np.random.seed(1)
    adj = np.random.random((n, n))
    features = np.random.random((n, f))

    y = np.random.randint(0, c, (n, ))
    labels = onehot(y, c)

    # prepare train_mask
    train_mask = sample_mask(range(2), n)
    val_mask = sample_mask(range(2, 5), n)
    test_mask = sample_mask(range(8, 10), n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data(dataset_str="cora"):
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

    # 2. get ordered features and labels
    index = parse_index_file("../data/ind.{}.test.index".format(dataset_str))  # 1708~2707 乱序
    index_sorted = np.sort(index)

    features = sp.vstack((allx, tx)).tolil()  # lil_matrix, todense  position
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

## gcn load

def gcn_load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


