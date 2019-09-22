import numpy as np
from layers.original_utils import onehot, sample_mask, softmax, numerical_grad, l2_loss
from layers.original_inits import init_Weight, init_dropout

# cross-entrocpy loss
def forward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """y_onehot: one_hot. train_mask: []"""
    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_real = np.multiply(cross_sum, train_mask.reshape(-1, 1))
    return np.mean(cross_real)


def backward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """require shape: outputs.shape"""
    m, n = outputs.shape
    dX = softmax(outputs) - y_onehot
    dX = np.multiply(dX, train_mask.reshape(-1, 1))
    return dX * (1 / (m*n))

# hidden
def act(X):
    return np.maximum(X, 0)

def backward_act(X):
    return np.where(X <= 0, 0, 1)


def forward_hidden(adj, hidden, weight_hidden, drop_out=0.5, drop_flag=False, bias_flag=False):
    # todo drop out;
    # if drop_flag:
    #     hidden *= init_dropout(hidden.shape, drop_out)
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)
    # todo add bias ? have to update
    # if bias_flag:
    #     A_tilde += np.zeros(A_tilde.shape)

    H = act(A_tilde)
    return H


def backward_hidden(adj, hidden, weight_hidden, pre_layer_grad):
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)

    dact = backward_act(A_tilde)
    dW = np.dot(A_hat.T, np.multiply(pre_layer_grad, dact))

    dAhat = np.dot(np.multiply(pre_layer_grad, dact), weight_hidden.T)
    dX = np.dot(adj.T, dAhat)
    return dX, dW


def test_gcn():
    n, f, h, c, l = 4, 2, 3, 3, 2
    # inputs inits
    inputs = np.random.random((n, f))
    adj = np.random.random((n, n))

    # hidden_x inits
    weights_hidden = init_Weight((f, h))

    # hidden inits
    weights_outputs = init_Weight((h, c))

    # loss inits
    y_true = np.random.randint(0, c, (n,))
    y_onehot = onehot(y_true, c)
    train_mask = sample_mask(range(l), n)

    # backwards
    hidden = forward_hidden(adj, inputs, weights_hidden)
    outputs = forward_hidden(adj, hidden, weights_outputs)

    grad_loss = backward_cross_entrocpy_loss(outputs, y_onehot, train_mask)
    grad_hidden, grad_weight_outputs = backward_hidden(adj, hidden, weights_outputs, grad_loss)
    _, grad_weight_hidden = backward_hidden(adj, inputs, weights_hidden, grad_hidden)

    grad_weight_hidden += 0.5 * weights_hidden

    # check grad
    loss_weight_hidden_f = lambda x: forward_cross_entrocpy_loss(
        forward_hidden(adj, forward_hidden(adj, inputs, x), weights_outputs), y_onehot, train_mask) + 0.5 * l2_loss(x)

    loss_weight_outputs_f = lambda x: forward_cross_entrocpy_loss(
        forward_hidden(adj, forward_hidden(adj, inputs, weights_hidden), x), y_onehot, train_mask) + 0.5 * l2_loss(weights_hidden)

    check_grad_weight_outputs = numerical_grad(loss_weight_outputs_f, weights_outputs)
    check_grad_weight_hidden = numerical_grad(loss_weight_hidden_f, weights_hidden)

    print("grad weight outputs", grad_weight_outputs)
    print("check grad weight outputs", check_grad_weight_outputs)

    print("grad weight hidden", grad_weight_hidden)
    print("check grad weight hidden", check_grad_weight_hidden)


if __name__ == '__main__':
    test_gcn()

