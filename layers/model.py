from layers.utils import load_data, l2_loss
from layers.inits import preprocess_adj, preprocess_features, preprocess_Delta
from layers.inits import init_dropout, init_Weight, masked_accuracy
from layers.lossLayer import forward_loss

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# 1. prepare data

# step 1. get original data
# adj--A,  features--X, labels--Y , train_mask, val_mask, test_mask
adj, features, labels, train_mask, val_mask, test_mask = load_data("cora")

# step 2. get preprocessed data

delta = preprocess_Delta(adj)

# a. get symmetrically normalize adj matrix  (position_list, value_list, shape)
adj = preprocess_adj(adj)   # every layer is the same
# b. row normalize features,  same to adj
features = preprocess_features(features)

# warp to placeholders
placeholders = {
    'features': features,
    'adj':  adj,
    'delta': delta,
    'dropout': 0.5,
    'labels_mask': train_mask,  # just for train
    'labels': labels  # onehot
}

# 2. intialize model

# W0,  W1
class Model(object):
    """model """
    def __init__(self):
        """init feathers, labels, input_dim"""
        # vars and placeholders
        self.vars = {}
        self.placeholders = {}

        # layers
        self.layers = []
        # activations
        self.activations = []

        # special input and output
        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _loss(self):
        """return loss, grad"""
        return NotImplementedError

    def _accuracy(self):
        """return accuracy"""
        return NotImplementedError

    def build(self):
        self._build()

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]
        self._loss()
        self._accuracy()
        # loss minimize todo adam
        #
        self.opt_op = self.optimizer.minimize(self.loss)  # todo

    def _build(self):
        """add layers"""
        return NotImplementedError

class GCN(Model):
    def __init__(self, placeholders, input_dim):
        super(GCN, self).__init__()
        self.inputs = placeholders['features']
        self.input_dim = input_dim  # can easily get by self.inputs
        self.output_dim = placeholders['labels'].shape[1]

        self.placeholders = placeholders

        self.build()

    def _loss(self):
        # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * l2_loss(var)

        # Cross entropy loss
        self.loss += forward_loss(self.outputs,
                                  self.placeholders['labels'],
                                  self.placeholders['delta'],
                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        # input_dim, output_dim, placeholders, act, dropput, sparse_inputs
        # attention! there are something else:
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=lambda x: np.maximum(x, 0),
                                            back_act=lambda x: np.where(x <= 0, 0, 1),
                                            dropout=True,
                                            sparse_inputs=False))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            back_act = lambda x: np.where(x == np.inf, 0, 1),
                                            dropout=True,
                                            sparse_inputs=False))

class Layer(object):
    def __init__(self):
        """something special: w1, sparse_flag,"""
        self.vars = {}
        self.sparse_inputs = False

    def _call(self, inputs):
        """return output? no layers"""
        # tensorflow just use numerical ways, so grad only needs _call. don't need backward.
        return inputs

class GraphConvolution(Layer):
    """Graph Convolution Layer"""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=lambda x: x,
                 back_act=lambda x: np.where(x == np.inf, 0, 1), bias=False, featureless = False):
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
        # if bias, init bias
        if self.bias:
            self.vars['bias'] = np.zeros([output_dim])

    def _call(self, inputs):
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

    def _back(self, dH):
        dact = self.back_act(self.hidden_tilde)
        # dW
        dW = np.dot(self.hidden_hat.T, dH * dact)
        # dX
        dhat = np.dot(dH * dact, self.vars['weight'].T)
        dX = np.dot(self.support.T, dhat)
        return dW, dX

class adam:
    pass

# 3. train


# train_loss, train_accuracy = model.fit(x_train, y_train)  update(w0, w1)
# val_loss, val_accuracy = model.evaluate(x_val, y_val)
# for epochs,  stop

# 4. test
# one time:  test_loss, test_accuracy = model.evalute(x_val, y_val)

