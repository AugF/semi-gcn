import scipy.sparse as sp
import pickle as pkl
import scipy.sparse as sp
import numpy as np
import networkx as nx
import sys
from layers.original_utils import onehot

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