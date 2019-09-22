import numpy as np
from layers.original_utils import onehot, softmax, numerical_grad, l2_loss
from layers.original_inits import init_Weight, init_dropout, Adam, masked_accuracy
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
    # use load_data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = prepare_gcn()
    n, f, c = adj.shape[0], features.shape[1], y_train.shape[1]
    h = 4

    features = features
    y_train = y_train

    # init weight
    weights_hidden = init_Weight((f, h))
    weights_outputs = init_Weight((h, c))

    # add adam train

    # init weights adam
    adam_weight_hidden = Adam(weights=weights_hidden, learning_rate=0.01)
    adam_weight_outputs = Adam(weights=weights_outputs, learning_rate=0.01)

    # train
    for i in range(20):
        weights_hidden = adam_weight_hidden.theta_t
        weights_outputs = adam_weight_outputs.theta_t

        hidden = forward_hidden(adj, features, weights_hidden)
        outputs = forward_hidden(adj, hidden, weights_outputs)
        loss = forward_cross_entrocpy_loss(outputs, y_train, train_mask)
        acc = masked_accuracy(outputs, y_train, train_mask)

        grad_loss = backward_cross_entrocpy_loss(outputs, y_train, train_mask)
        grad_hidden, grad_weight_outputs = backward_hidden(adj, hidden, weights_outputs, grad_loss)

        _, grad_weight_hidden = backward_hidden(adj, features, weights_hidden, grad_hidden)
        grad_weight_hidden += 0.5 * weights_hidden

        adam_weight_hidden.minimize(grad_weight_hidden)
        adam_weight_outputs.minimize(grad_weight_outputs)

        print("iteration: {}, loss: {}, acc: {}".format(i, loss, acc))
        # print("weight_hidden", adam_weight_hidden.theta_t)
        # print("weight_ouputs", adam_weight_outputs.theta_t)


if __name__ == '__main__':
    test_gcn()

