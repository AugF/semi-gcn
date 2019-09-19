import numpy as np

"""2 types of loss"""
# cross entrocpy loss

def softmax(X):
    exp_x = np.exp(X)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    softmax_x = exp_x / sum_x
    return softmax_x


def masked_softmax_cross_entropy_loss(X, Y, train_mask):
    """todo: check there is a mean, so gradient should multiply a constant '1 / mn'"""
    softmax_x = softmax(X)

    # print("exist x[i, j] == 0 ? ", np.where(softmax_x == 0, 1, 0).any())
    # print("exist x[i, j] == np.inf? ", np.where(softmax_x == np.inf, 1, 0).any())
    # print("exist x[i, j] == -np.inf? ", np.where(softmax_x == np.inf, 1, 0).any())
    # print("exist x[i, j] < 0? ", np.where(softmax_x < 0, 1, 0).any())

    cross_sum = np.multiply(np.log(softmax_x), -Y)   # -y * log x
    cross_real = np.multiply(cross_sum, train_mask.reshape(-1, 1))
    return np.mean(cross_real)


def masked_softmax_backward(X, Y, train_mask):
    """p is the constant"""
    p = 1.0 / (X.shape[0] * X.shape[1])
    dX = softmax(X) - Y
    dX = np.multiply(dX, train_mask.reshape(-1, 1))
    return p*dX

# reg loss
def regloss(x, A):
    """x: (1, 2708)"""
    loss = np.dot(np.dot(x, A), x.T)
    return loss.tolist()[0][0]


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
    # p = np.array(x.tolist()[0])
    return x


def backward_regloss(x, A):
    """x: (1,2708)"""
    return np.dot((A + A.T), x.T)


def backward_argsoftmax(X, beta):
    # require n, c
    n, c = X.shape
    softmax_x = softmax_beta(X, beta=beta)  # beta * (j - yi) * softmax_x

    index = np.array(range(1, c + 1))
    x = np.dot(softmax_x, index)
    j_yi = np.repeat(index.reshape(1, -1), n, axis=0) - x.reshape(-1, 1)                               # j-yi

    grad = beta * np.multiply(softmax_x, j_yi)
    return grad


def forward_loss(X, Y, A, train_mask):
    loss = masked_softmax_cross_entropy_loss(softmax(X), Y, train_mask)
    print("loss", loss)
    loss_reg = regloss(argsoftmax(X, beta=10), A)
    return 1e-10 * loss_reg + loss


def backward_grad(X, Y, A, train_mask):
    # loss
    grad_1 = masked_softmax_backward(X, Y, train_mask)

    # reg
    fx = argsoftmax(X, beta=10)
    grad_reg = backward_regloss(fx, A)
    grad_softargmax = backward_argsoftmax(X, 10)

    grad_2 = np.multiply(grad_softargmax, grad_reg.reshape(-1, 1))

    grad = grad_1 + grad_2
    return grad
