import numpy as np


def onehot(y, c):
    res = np.zeros((len(y), c))
    for i, j in enumerate(y):
        res[i, j] = 1
    return res


def prepare_gcn():
    n, f, h, c = 4, 4, 2, 3
    np.random.seed(1)
    A = np.random.random((n, n))  #
    P = np.random.random((n, n))  # reg
    X = np.random.random((n, f)) # features

    W0 = np.random.random((f, h))
    W1 = np.random.random((h, c))

    y = np.random.randint(0, c, (n, ))
    Y = onehot(y, c)
    return A, P, X, W0, W1, Y


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
