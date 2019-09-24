import numpy as np
from layers.original_inits import *

# hidden


def forward_hidden(adj, hidden, weight_hidden, act=lambda x: x, drop_out=0.5, drop_flag=False):
    # todo drop out;
    if drop_flag:
        hidden = np.multiply(hidden, init_dropout(hidden.shape, 1 - drop_out))
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)

    H = act(A_tilde)
    return H


def backward_hidden(adj, hidden, weight_hidden, pre_layer_grad, backward_act=lambda x: np.ones(x.shape)):
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)

    dact = backward_act(A_tilde)
    dW = np.dot(A_hat.T, np.multiply(pre_layer_grad, dact))

    dAhat = np.dot(np.multiply(pre_layer_grad, dact), weight_hidden.T)
    dX = np.dot(adj.T, dAhat)
    return dX, dW



