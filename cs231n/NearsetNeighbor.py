import numpy as np

class NearstNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """X is N*D where each row is an example. Y is 1-dimension of size N"""
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """X is N*D where each row is an example we wish to predict label for"""
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred


def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def L(x, y, z):
    pass

import autograd

bestloss = float("inf")
for num in range(1000):
    W = np.random.rand(10, 3073)*0.0001  # why need to 0.0001
    X_train = Y_train = W = 0
    loss = L(X_train, Y_train , W)
    if loss < bestloss:
        bestloss = loss
        bestW = W
    print("in attempt %d the loss was %f, best %f" % (num, loss, bestloss))


# Vanilla Gradient Descent
def evaluate_gradient(a, b, c):
    pass

loss_fun = lambda x,y:x
data = 0
weights = 0
step_size = 0

while True:
    weights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += - step_size * weights_grad


class ComputationGraph(object):
    # ...
    def forward(self, inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph
        for gate in self.graph.nodes_topologically_sorted():
            gate.forward()
        return loss

    def backward(self):
        for gate in reversed(self.graph.nodes_topologically_sorted):
            gate.backward()
        return inputs_gradients


class MultipyGate(object):
    """x, y,z: scalars"""
    def forward(self, x, y):
        z = x*y
        self.x = x
        self.y = y
        return z

    def backward(self, dz):
        dx = self.y * dz # [dz/dx * dL/dz]
        dy = self.x * dz # [dz/dy * dL/dz]
        return [dx, dy]

# receive W (weights), X (data)
# forward pass (we have 5 lines)
scores = 1   # f=Wx
margins = 1
data_loss = 1
reg_loss = 1
loss = data_loss + reg_loss

# backward pass ( we have 5 lines)
dmargins = 1  # optionally, we go direct to dscores
dscores = 1
dW = 1





