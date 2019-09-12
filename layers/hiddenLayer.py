import numpy as np
from layers.lossLayer import forward_loss, backward_grad, onehot


def act(X):
    return np.maximum(X, 0)


def backward_act(X):
    return np.where(X <= 0, 0, 1)


def forward_hidden(X, A, W):
    A_hat = np.dot(A, X)
    A_tilde = np.dot(A_hat, W)
    H = act(A_tilde)
    return H, A_tilde, A_hat


def backward_hidden(A, W, dH, A_tilde, A_hat):
    dact = backward_act(A_tilde)
    dW = np.dot(A_hat.T, dH * dact)

    dAhat = np.dot(dH * dact, W.T)
    dX = np.dot(A.T, dAhat)
    return dW, dX


def fun():
    # require H--(n, c)  before softmax
    # X - (n, f）  W - (f, c)
    n, f, c = 4, 2, 3
    np.random.seed(1)
    A = np.random.random((n, n))
    P = np.random.random((n, n))
    X = np.random.random((n, f))
    W = np.random.random((f, c))

    y = np.random.randint(0, c, (n, ))
    Y = onehot(y, c)

    # 1. forward
    # A_tilde = AXW
    A_hat = np.dot(A, X)
    A_tilde = np.dot(A_hat, W)

    # act
    H = act(A_tilde)
    print(H.shape)

    # loss
    loss = forward_loss(H, Y, P)

    # 2. backward
    dH = backward_grad(H, Y, P)  # n, c

    dact = backward_act(A_tilde)  # n, c

    # dW = np.dot(A_hat.T, dH * dact) # f, c

    dAhat = np.dot(dH * dact, W.T)
    dX = np.dot(A.T, dAhat)  # n*f

    print("grad", dX)
    # 3. check_grad
    h = 1e-5

    # 3-1. dW
    # check_grad = np.zeros(dW.shape)
    #
    # for i in range(f):
    #     for j in range(c):
    #         W[i, j] += h
    #         loss1 = forward_loss(np.dot(A_hat, W), Y, P)
    #         W[i, j] -= 2*h
    #         loss2 = forward_loss(np.dot(A_hat, W), Y, P)
    #         W[i, j] += h
    #         check_grad[i, j] = (loss1 - loss2) / (2*h)
    # print("check_grad", check_grad)

    # 3-2. dX
    check_grad = np.zeros(dX.shape)

    for i in range(n):
        for j in range(f):
            X[i, j] += h
            loss1 = forward_loss(np.dot(np.dot(A, X), W), Y, P)
            X[i, j] -= 2*h
            loss2 = forward_loss(np.dot(np.dot(A, X), W), Y, P)
            X[i, j] += h
            check_grad[i, j] = (loss1 - loss2) / (2*h)
    print("check_grad", check_grad)

if __name__ == '__main__':
    fun()