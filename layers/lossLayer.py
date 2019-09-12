from layers.softmaxloss import softmax, _loss, backward
from layers.softmax_regularize import reg_loss, argsoftmax, backward_regloss, backward_argsoftmax
from layers.utils import onehot, sample_mask
import numpy as np

def forward_loss(X, Y, A, train_mask):
    loss = _loss(softmax(X), Y, train_mask)
    loss_reg = reg_loss(argsoftmax(X, beta=10), A)
    return loss + loss_reg

def backward_grad(X, Y, A, train_mask):
    # loss
    grad_1 = backward(X, Y, train_mask)

    # reg
    fx = argsoftmax(X, beta=10)
    grad_reg = backward_regloss(fx, A)
    grad_softargmax = backward_argsoftmax(X, 10)

    grad_2 = grad_softargmax * grad_reg.reshape(-1, 1)

    grad = grad_1 + grad_2
    return grad

def check_loss():
    n, c = 4, 3
    np.random.seed(1)
    X = np.random.random((n, c))
    A = np.random.random((n, n))
    y = np.random.randint(0, c, (n,))
    Y = onehot(y, c)

    l = 2
    train_mask = sample_mask(range(l), n)

    # forward
    loss = forward_loss(X, Y, A, train_mask)

    # backward
    grad = backward_grad(X, Y, A, train_mask)

    print("grad", grad)

    # check grad
    check_grad = np.zeros(grad.shape)

    h = 1e-5
    for i in range(n):
        for j in range(c):
            X[i, j] += h
            loss1 = forward_loss(X, Y, A, train_mask)
            X[i, j] -= 2*h
            loss2 = forward_loss(X, Y, A, train_mask)
            check_grad[i, j] = (loss1 - loss2) / (2*h)
            X[i, j] += h

    print("check_grad", check_grad)

if __name__ == '__main__':
    check_loss()
