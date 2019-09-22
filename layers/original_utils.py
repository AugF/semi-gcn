import numpy as np


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


def onehot(y, c):
    res = np.zeros((len(y), c))
    for i, j in enumerate(y):
        res[i, j] = 1
    return res


def sample_mask(idx, l):
    # idx: sample_list;   l: total length
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def l2_loss(X):
    """for matrix, tf.nn.l2_loss: np.sum(x**2)/2"""
    x_square = X ** 2
    x_sum = np.sum(x_square)
    x_l2 = x_sum / 2
    return x_l2

def softmax(X):
    """softmax """
    exp_x = np.exp(X)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    softmax_x = exp_x / sum_x
    return softmax_x


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
