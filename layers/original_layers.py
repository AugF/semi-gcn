import numpy as np
from layers.original_utils import onehot, softmax, numerical_grad, l2_loss
from layers.original_inits import init_Weight, init_dropout, Adam, masked_accuracy, preprocess_features, preprocess_adj
from layers.original_load import load_data, sample_mask, prepare_gcn

# cross-entrocpy loss
def forward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """y_onehot: one_hot. train_mask: []"""
    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_sum = np.sum(cross_sum, axis=1)
    cross_real = np.multiply(cross_sum, train_mask)
    return np.mean(cross_real)

def backward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """require shape: outputs.shape"""
    dX = softmax(outputs) - y_onehot
    dX = np.multiply(dX, train_mask.reshape(-1, 1))
    return dX / outputs.shape[0]

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



