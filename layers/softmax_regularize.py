import numpy as np
from gcn.utils import numerical_grad


def reg_loss(x, A):
    loss = np.dot(np.dot(x.T, A), x)
    return loss


def softmax_beta(X, beta):
    exp_x = np.exp(beta * X)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    softmax_beta_x = exp_x / sum_x
    return softmax_beta_x


def argsoftmax(X, beta):
    softmax_beta_x = softmax_beta(X, beta)

    # argsoftmax
    index = np.array(range(1, X.shape[1] + 1))
    x = np.dot(softmax_beta_x, index)
    return x

def backward_regloss(x, A):
    return np.dot((A + A.T), x)

def backward_argsoftmax(X, beta):
    # require n, c
    n, c = X.shape
    softmax_x = softmax_beta(X, beta=beta)  # beta * (j - yi) * softmax_x

    index = np.array(range(1, c + 1))
    x = np.dot(softmax_x, index)
    j_yi = np.repeat(index.reshape(1, -1), n, axis=0) - x.reshape(-1, 1)                               # j-yi

    grad = beta * softmax_x * j_yi
    return grad

def main():
    n, c = 4, 3
    np.random.seed(1)
    X = np.random.random((n, c))
    A = np.random.random((n, n))

    softmax_x = softmax_beta(X, 1)
    # loss
    fx = argsoftmax(X, beta=10)
    fx_real = np.argmax(X, axis=1)
    # print("X", X)
    # print("A", A)
    # print(softmax_x)
    # print(fx)
    # print(fx_real)

    loss = reg_loss(fx, A)
    print(loss)

    h = 1e-5
    # check reg
    grad_reg = backward_regloss(fx, A)
    # print("grad reg", grad_reg)
    #
    # check_reg = np.zeros(grad_reg.shape)
    # for i in range(n):
    #     fx[i] += h
    #     loss1 = reg_loss(fx, A)
    #     fx[i] -= 2*h
    #     loss2 = reg_loss(fx, A)
    #     fx[i] += h
    #     check_reg[i] = (loss1 - loss2) / (2*h)
    # print("check_reg", check_reg)

    # check softargmax + reg
    grad_softargmax = backward_argsoftmax(X, 10)
    grad = grad_softargmax * grad_reg.reshape(-1, 1)
    print("grad", grad)

    f = lambda x: reg_loss(argsoftmax(X, beta=10), A)

    check_grad = numerical_grad(f, X)

    # check_grad = np.zeros(grad.shape)
    #
    # for i in range(n):
    #     for j in range(c):
    #         X[i, j] += h
    #         loss1 = reg_loss(argsoftmax(X, beta=10), A)
    #         X[i, j] -= 2*h
    #         loss2 = reg_loss(argsoftmax(X, beta=10), A)
    #         check_grad[i, j] = (loss1 - loss2) / (2*h)
    #         X[i, j] += h
    
    print("check_grad", check_grad)


if __name__ == '__main__':
    main()
