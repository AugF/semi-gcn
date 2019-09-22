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
def act(X):
    return np.maximum(X, 0)

def backward_act(X):
    return np.where(X <= 0, 0, 1)


def forward_hidden(adj, hidden, weight_hidden, act=lambda x: x, drop_out=0.5, drop_flag=False, bias_flag=False):
    # todo drop out;
    if drop_flag:
        hidden = np.multiply(hidden, init_dropout(hidden.shape, 1 - drop_out))
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)
    # todo add bias ? have to update
    # if bias_flag:
    #     A_tilde += np.zeros(A_tilde.shape)

    H = act(A_tilde)
    return H


def backward_hidden(adj, hidden, weight_hidden, pre_layer_grad, back_act=lambda x: np.ones(x.shape)):  # todo change
    A_hat = np.dot(adj, hidden)
    A_tilde = np.dot(A_hat, weight_hidden)

    dact = backward_act(A_tilde)
    dW = np.dot(A_hat.T, np.multiply(pre_layer_grad, dact))

    dAhat = np.dot(np.multiply(pre_layer_grad, dact), weight_hidden.T)
    dX = np.dot(adj.T, dAhat)
    return dX, dW


class GCN:
    """GCN"""
    def __init__(self, load_data_function, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_function(0)
        self.adj, self.features = preprocess_adj(adj), preprocess_features(features)  # preprocess

        self.y_train, self.train_mask = y_train, train_mask
        self.y_val, self.val_mask = y_val, val_mask
        self.y_test, self.test_mask = y_test, test_mask
        # init
        self.n, self.f, self.c = adj.shape[0], features.shape[1], y_train.shape[1]
        self.h = hidden_unit
        # init weight
        self.weight_hidden = init_Weight((self.f, self.h))
        self.weight_outputs = init_Weight((self.h, self.c))

        self.adam_weight_hidden = Adam(weights=self.weight_hidden, learning_rate=learning_rate)
        self.adam_weight_outputs = Adam(weights=self.weight_outputs, learning_rate=learning_rate)

        self.hidden = np.zeros((self.n, self.h))
        self.outputs = np.zeros((self.n, self.c))

    def evaluate(self):
        y_train, train_mask = self.y_val, self.val_mask
        hidden = forward_hidden(self.adj, self.features, self.weight_hidden)
        outputs = forward_hidden(self.adj, hidden, self.weight_outputs)
        loss = forward_cross_entrocpy_loss(outputs, y_train, train_mask)
        acc = masked_accuracy(outputs, y_train, train_mask)
        return loss, acc

    def test(self):
        y_train, train_mask = self.y_test, self.test_mask
        hidden = forward_hidden(self.adj, self.features, self.weight_hidden)
        outputs = forward_hidden(self.adj, hidden, self.weight_outputs)
        loss = forward_cross_entrocpy_loss(outputs, y_train, train_mask)
        acc = masked_accuracy(outputs, y_train, train_mask)
        return loss, acc

    def one_train(self):
        y_train, train_mask = self.y_train, self.train_mask
        self.hidden = forward_hidden(self.adj, self.features, self.weight_hidden)
        self.outputs = forward_hidden(self.adj, self.hidden, self.weight_outputs)
        loss = forward_cross_entrocpy_loss(self.outputs, y_train, train_mask)
        acc = masked_accuracy(self.outputs, y_train, train_mask)
        return loss, acc

    def one_update(self):
        y_train, train_mask = self.y_train, self.train_mask
        grad_loss = backward_cross_entrocpy_loss(self.outputs, y_train, train_mask)
        grad_hidden, grad_weight_outputs = backward_hidden(self.adj, self.hidden, self.weight_outputs, grad_loss)

        _, grad_weight_hidden = backward_hidden(self.adj, self.features, self.weight_hidden, grad_hidden)
        grad_weight_hidden += 0.5 * self.weight_hidden

        self.adam_weight_hidden.minimize(grad_weight_hidden)
        self.adam_weight_outputs.minimize(grad_weight_outputs)

        self.weight_hidden = self.adam_weight_hidden.theta_t
        self.weight_outputs = self.adam_weight_outputs.theta_t



def test_gcn(early_stopping=10, ephochs=200, data_str="cora", dropout=0.5, hidden_unit=16, learning_rate=0.01, weight_decay=5e-4):
    load_data_function = lambda x: load_data(data_str)
    model = GCN(load_data_function=load_data_function, hidden_unit=hidden_unit, learning_rate=learning_rate, weight_decay=weight_decay)

    cost_val = []
    # train
    for i in range(ephochs):
        # train step
        train_loss, train_acc = model.one_train()
        model.one_update()

        # val step
        val_loss, val_acc = model.evaluate()
        print("iteration: {}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}".
              format(i, train_loss, train_acc, val_loss, val_acc))
        cost_val.append(val_loss)

        if i > early_stopping and cost_val[-1] > np.mean(cost_val[-(early_stopping + 1): -1]):
            print("early stopping ! ")

    test_loss, test_acc = model.test()
    print("start test, the loss: {}, the acc: {}".format(test_loss, test_acc))

if __name__ == '__main__':
    test_gcn()

