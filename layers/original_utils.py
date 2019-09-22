import numpy as np

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

def softmax(X):
    """softmax x"""
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
