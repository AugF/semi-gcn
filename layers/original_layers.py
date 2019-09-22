import numpy as np
from layers.original_utils import onehot, sample_mask, softmax, numerical_grad

# cross-entrocpy loss
def forward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """y_onehot: one_hot. train_mask: []"""
    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_real = np.multiply(cross_sum, train_mask.reshape(-1, 1))
    return np.sum(cross_real)


def backward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """require shape: outputs.shape"""
    dX = softmax(outputs) - y_onehot
    dX = np.multiply(dX, train_mask.reshape(-1, 1))
    return dX

# hidden
def act(X):
    return np.maximum(X, 0)

def backward_act(X):
    return np.where(X <= 0, 0, 1)


def forward_hidden(adj, hidden, weight_hidden):
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)
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


def test_hidden():
    n, f, c, l = 4, 2, 3, 2
    # hidden inits
    adj = np.random.random((n, n))
    hidden = np.random.random((n, f))
    weights_hidden = np.random.random((f, c))

    # loss inits
    y_true = np.random.randint(0, c, (n,))
    y_onehot = onehot(y_true, c)
    train_mask = sample_mask(range(l), n)

    # backward
    outputs = forward_hidden(adj, hidden, weights_hidden)
    pre_layer_grad = backward_cross_entrocpy_loss(outputs, y_onehot, train_mask)
    grad_hidden, grad_weight = backward_hidden(adj, hidden, weights_hidden, pre_layer_grad)

    loss_weight_f = lambda x: forward_cross_entrocpy_loss(forward_hidden(adj, hidden, x), y_onehot, train_mask)
    loss_hidden_f = lambda x: forward_cross_entrocpy_loss(forward_hidden(adj, x, weights_hidden), y_onehot, train_mask)

    check_grad_weight = numerical_grad(loss_weight_f, weights_hidden)
    check_grad_hidden = numerical_grad(loss_hidden_f, hidden)

    print("grad_weight", grad_weight)
    print("check_grad_weight", check_grad_weight)

    print("grad hidden", grad_hidden)
    print("check_grad_hidden", check_grad_hidden)

def test_gcn():
    n, f, h, c, l = 4, 2, 3, 3, 2
    # inputs inits
    inputs = np.random.random((n, f))
    adj = np.random.random((n, n))

    # hidden_x inits
    weights_hidden = np.random.random((f, h))

    # hidden inits
    weights_outputs = np.random.random((h, c))

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

    # check grad
    loss_weight_hidden_f = lambda x : forward_cross_entrocpy_loss(
        forward_hidden(adj, forward_hidden(adj, inputs, x), weights_outputs), y_onehot, train_mask)

    loss_weight_outputs_f = lambda x: forward_cross_entrocpy_loss(
        forward_hidden(adj, forward_hidden(adj, inputs, weights_hidden), x), y_onehot, train_mask)

    check_grad_weight_outputs = numerical_grad(loss_weight_outputs_f, weights_outputs)
    check_grad_weight_hidden = numerical_grad(loss_weight_hidden_f, weights_hidden)

    print("grad weight outputs", grad_weight_outputs)
    print("check grad weight outputs", check_grad_weight_outputs)

    print("grad weight hidden", grad_weight_hidden)
    print("check grad weight hidden", check_grad_weight_hidden)


def test_cross_entrocpy_loss():
    n, c = 4, 3
    outputs = np.random.random((n, c))
    y_true = np.random.randint(0, c, (n,))
    y_onehot = onehot(y_true, c)

    l = 2
    train_mask = sample_mask(range(l), n)

    # grad
    grad = backward_cross_entrocpy_loss(outputs, y_onehot, train_mask)

    # check-grad
    g = lambda x: forward_cross_entrocpy_loss(x, y_onehot, train_mask)  # forward
    check_grad = numerical_grad(g, outputs)
    print("grad", grad)
    print("check_grad", check_grad)


if __name__ == '__main__':
    test_gcn()

