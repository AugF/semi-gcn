import numpy as np

def onehot(y, c):
    res = np.zeros((len(y), c))
    for i, j in enumerate(y):
        res[i, j] = 1
    return res

def softmax(X):
    exp_x = np.exp(X)
    sum_x = np.sum(exp_x, axis=1).reshape(-1, 1)
    softmax_x = exp_x / sum_x
    return softmax_x

def _loss(X, Y):
    return np.sum(-Y * np.log(X))

def backward(X, Y):
    dX = softmax(X) - Y
    return dX

def main():
    n, c = 4, 3
    X = np.random.random((n, c))
    y = np.random.randint(0, c, (n,))
    Y = onehot(y, c)

    # forward
    softmax_X = softmax(X)
    loss = _loss(softmax_X, Y)

    print(loss)
    # grad
    grad = backward(X, Y)

    # check-grad
    h = 1e-5
    X_copy = np.copy(X)
    check_grad = np.zeros((n, c))
    for i in range(n):
        for j in range(c):
            X_copy[i, j] += h
            loss1 = _loss(softmax_X(X_copy), Y)
            X_copy[i, j] -= 2 * h
            loss2 = _loss(softmax_X(X_copy), Y)
            check_grad[i, j] = (loss2 - loss1) / (2*h)
            # recover
            X_copy[i, j] += h

    print(grad)
    print(check_grad)


class SoftmaxLossLayer(object):
    """ forward,  backward"""
    def __init__(self, X, y, labeled):
        self.X = X
        self.y = y
        self.labeled = labeled

    def forward(self):
        # 1. stable x
        # x = x - np.max(x, axis=1).reshape(-1, 1)
        # 2. softmax
        X, y, labeled = self.X, self.y, self.labeled
        exp_x = np.exp(X)
        sum_x = np.sum(exp_x, axis=1)
        softmax_x = exp_x / sum_x
        # 3. loss
        loss = 0
        for i in range(labeled):
            loss -= np.log(softmax_x[i, y[i]])

        self.softmax_x , self.loss = softmax_x, loss
        return loss

    def backward(self):
        softmax_x, X, y = self.softmax_x, self.X, self.y
        # 1. get one-hot
        Y = np.empty(X.shape)
        for i in y:
            Y[i] = 1
        # 2. dX
        dX = softmax_x - Y  # pi - yi
        self.dX = dX
        return dX

    def grad_check(self):
        h = 1e-5



if __name__ == '__main__':
    main()
