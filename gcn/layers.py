import numpy as np
from gcn.inits import init_dropout, init_Weight
from gcn.adam import Adam

class Layer(object):
    """Normal layer"""
    def __init__(self):
        """something special: w1, sparse_flag,"""
        self.vars = {}
        self.adam = None
        self.sparse_inputs = False

    def call(self, inputs):
        """return output? no layers"""
        # tensorflow just use numerical ways, so grad only needs _call. don't need backward.
        return NotImplementedError

    def back(self, grad_pre_layer):
        return NotImplementedError


class GraphConvolution(Layer):
    """Graph Convolution Layer"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=lambda x: x,
                 back_act=lambda x: np.where(x == np.inf, 0, 1), bias=False, featureless=False):
        """init function"""
        super(GraphConvolution, self).__init__()

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.back_act = back_act
        self.support = placeholders['adj']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless  # ??
        self.bias = bias

        # init weight
        self.vars['weight'] = init_Weight([input_dim, output_dim])

        # init adam
        self.adam = Adam(weights=self.vars['weight'])

        # if bias, init bias
        if self.bias:
            self.vars['bias'] = np.zeros([output_dim])

    def call(self, inputs):
        """call: forward function"""
        x = inputs

        # drop put
        if self.sparse_inputs:
            pass  # todo finish sp
        else:
            x = init_dropout(inputs.shape, self.dropout)

        # convolve  what is featureless
        self.hidden_hat = np.dot(self.support, x)
        self.hidden_tilde = np.dot(self.hidden_hat, self.vars['weight'])

        # bias
        if self.bias:
            self.hidden_tilde += self.vars['bias']
        return self.act(self.hidden_tilde)

    def back(self, grad_pre_layer):
        """back: backward"""
        grad_act = self.back_act(self.hidden_tilde)
        # dW
        grad_weight = np.dot(self.hidden_hat.T, np.multiply(grad_pre_layer, grad_act))
        # dX  todo only there is about weight
        grad_temp = np.dot(np.multiply(grad_pre_layer, grad_act), self.vars['weight'].T)
        grad_input = np.dot(self.support.T, grad_temp)
        return grad_weight, grad_input
