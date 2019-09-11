#  ex3:  a example
import numpy as np
from numpy.random import randn

N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2


# 3.5.
import math
class Neuron:
    # ..
    def neuron_tick(self, inputs):
        """assume inputs and weights are 1-D numpy arrays and bias is a number"""
        cell_body_sum = np.sum(inputs * self.weights) + self.bias
        firing_rate = 1.0 / (1.0 + math.exp(-cell_body_sum))
        return firing_rate



class TwoLayerNet(object):
    """
    affine - relu - affine - softmax
    """
    def __init__(self, input_dim, hidden_dim, num_classes, weight_scale, reg):
        self.param["w0"] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.param['b1'] = np.zeros(hidden_dim)
        self.param['w1'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.param['b1'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        计算一个batch的数据的loss和gradient
        :param X:
        :param y:
        :return:
        """
        scores = None

        # TODO 计算两层神经网络的froward pass
        # 计算 scores

        # affine_relu_out, affine_relu_cache =


