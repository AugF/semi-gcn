from layers.original_layers import *
from layers.original_metrics import *
from layers.original_inits import *
from layers.original_utils import *
from layers.original_optimizer import Adam

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
        # self.weight_hidden = init_Weight((self.f, self.h))
        # self.weight_outputs = init_Weight((self.h, self.c))
        self.weight_hidden = get_Weight_from_file("weights_hidden")
        self.weight_outputs = get_Weight_from_file("weights_outputs")

        self.adam_weight_hidden = Adam(weights=self.weight_hidden, learning_rate=learning_rate)
        self.adam_weight_outputs = Adam(weights=self.weight_outputs, learning_rate=learning_rate)

        self.hidden = np.zeros((self.n, self.h))
        self.outputs = np.zeros((self.n, self.c))

        self.weight_decay = weight_decay


    def evaluate(self):
        y_train, train_mask = self.y_val, self.val_mask
        hidden = forward_hidden(self.adj, self.features, self.weight_hidden, act=lambda x: np.maximum(x, 0))
        outputs = forward_hidden(self.adj, hidden, self.weight_outputs)
        loss = forward_cross_entrocpy_loss(outputs, y_train, train_mask)
        loss += self.weight_decay * l2_loss(self.weight_hidden)
        acc = masked_accuracy(outputs, y_train, train_mask)
        return loss, acc

    def test(self):
        y_train, train_mask = self.y_test, self.test_mask
        hidden = forward_hidden(self.adj, self.features, self.weight_hidden, act=lambda x: np.maximum(x, 0))
        outputs = forward_hidden(self.adj, hidden, self.weight_outputs)
        loss = forward_cross_entrocpy_loss(outputs, y_train, train_mask)
        loss = self.weight_decay * l2_loss(self.weight_hidden)
        acc = masked_accuracy(outputs, y_train, train_mask)
        return loss, acc

    def one_train(self):
        self.hidden = forward_hidden(self.adj, self.features, self.weight_hidden, act=lambda x: np.maximum(x, 0))  # the first hidden
        self.outputs = forward_hidden(self.adj, self.hidden, self.weight_outputs)
        # test
        loss = forward_cross_entrocpy_loss(self.outputs, self.y_train, self.train_mask)
        weight_decay_loss = self.weight_decay * l2_loss(self.weight_hidden)
        acc = masked_accuracy(self.outputs, self.y_train, self.train_mask)
        return loss, acc, weight_decay_loss

    def one_update(self):
        y_train, train_mask = self.y_train, self.train_mask
        grad_loss = backward_cross_entrocpy_loss(self.outputs, y_train, train_mask)
        grad_hidden, grad_weight_outputs = backward_hidden(self.adj, self.hidden, self.weight_outputs, grad_loss)

        _, grad_weight_hidden = backward_hidden(self.adj, self.features, self.weight_hidden, grad_hidden, backward_act=lambda x: np.where(x <= 0, 0, 1))
        grad_weight_hidden += self.weight_decay * self.weight_hidden  # weight_decay backward

        self.adam_weight_hidden.minimize(grad_weight_hidden)
        self.adam_weight_outputs.minimize(grad_weight_outputs)

        self.weight_hidden = self.adam_weight_hidden.theta_t
        self.weight_outputs = self.adam_weight_outputs.theta_t