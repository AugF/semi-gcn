import numpy as np
from layers.utils import onehot, sample_mask


def softmax(X):
    exp_x = np.exp(X)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    softmax_x = exp_x / sum_x
    return softmax_x

def _loss(X, Y, train_mask):
    cross_sum = -Y * np.log(X)
    cross_real = cross_sum * train_mask.reshape(-1, 1)
    return np.sum(cross_real)

def backward(X, Y, train_mask):
    dX = softmax(X) - Y
    dX = dX * train_mask.reshape(-1, 1)
    return dX

def main():
    n, c = 4, 3
    X = np.random.random((n, c))
    y = np.random.randint(0, c, (n,))
    Y = onehot(y, c)

    l = 2
    train_mask = sample_mask(range(l), n)

    # forward
    softmax_X = softmax(X)
    loss = _loss(softmax_X, Y, train_mask)

    print(loss)
    # grad
    grad = backward(X, Y, train_mask)

    # check-grad
    h = 1e-5
    X_copy = np.copy(X)
    check_grad = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            X_copy[i, j] += h
            loss1 = _loss(softmax(X_copy), Y, train_mask)
            X_copy[i, j] -= 2 * h
            loss2 = _loss(softmax(X_copy), Y, train_mask)
            check_grad[i, j] = (loss1 - loss2) / (2*h)
            # recover
            X_copy[i, j] += h

    print(grad)
    print(check_grad)

if __name__ == '__main__':
    main()
