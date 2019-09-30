from original.weights_preprocess import *

# hidden

def forward_hidden(adj, hidden, weight_hidden, act=lambda x: x, drop_out=0.5, drop_flag=False):
    """loss the dropout"""
    # todo dropout
    #  1. dense: tf.nn.dropout() bionaimal distribution not exisit
    #  2. sparse: random_uniform(noise_shape)  np has it!

    # todo sparse
    #  1. dropout
    #  2. multipy:  tf.sparse_tensor_dense_matmul(sp_a, b)
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)

    H = act(A_tilde)
    return H


def backward_hidden(adj, hidden, weight_hidden, pre_layer_grad, mask, backward_act=lambda x: np.ones(x.shape), mask_flag=False):
    """hidden: inputs for layers;   pre_layer_grad: loss about outputs of layers"""
    A_hat = np.dot(adj, hidden)

    if mask_flag:
        A_hat = np.multiply(A_hat, mask.reshape(-1, 1))  # only use the ordered data

    A_tilde = np.dot(A_hat, weight_hidden)

    dact = backward_act(A_tilde)
    dW = np.dot(A_hat.T, np.multiply(pre_layer_grad, dact))

    dAhat = np.dot(np.multiply(pre_layer_grad, dact), weight_hidden.T)
    dX = np.dot(adj.T, dAhat)
    return dX, dW
