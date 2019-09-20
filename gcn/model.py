from gcn.inits import masked_accuracy
from gcn.layers import GraphConvolution
from gcn.loss import masked_softmax_cross_entropy_loss, masked_softmax_backward
from gcn.utils import l2_loss

import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


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

    def _loss(self):
        """return loss, grad"""
        return NotImplementedError

    def _accuracy(self):
        """return accuracy"""
        return NotImplementedError

    def _backgrad(self):
        """back grad"""
        return NotImplementedError

    def build(self):
        self._build()

        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])  # the last element of current activations list
            self.activations.append(hidden)

        self.outputs = self.activations[-1]

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
        # Weight decay loss todo
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * l2_loss(var)

        # print("weight decay loss", self.loss)
        # Cross entropy loss
        self.loss += masked_softmax_cross_entropy_loss(self.outputs,
                                                       self.placeholders['labels'],
                                                       self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _backgrad(self):
        # update the gradient
        grad_pre_layer = masked_softmax_backward(self.outputs,
                                                 self.placeholders['labels'],
                                                 self.placeholders['labels_mask'])
        # update every layer
        for layer in reversed(self.layers):
            grad_weight, grad_pre_layer = layer.back(grad_pre_layer)  # weight
            layer.vars['weight'] -= 0.00001 * grad_weight
            # layer.vars['weight'] = layer.adam.minimize(grad_weight)  # adam

    def check_grad(self, weights):
        # check grad is true
        h = 1e-8
        grad_weights = np.zeros(weights.shape)
        for i in weights.shape[0]:
            for j in weights.shape[1]:
                grad_weights[i, j] += h
                grad_weights[i, j] -= 2*h

    def _forward(self):
        for layer in self.layers:
            layer.call()

    def one_train(self):
        self._loss()
        self._accuracy()
        self._backgrad()
        return self.loss, self.accuracy

    def evaluate(self, y_val, val_mask):
        self.placeholders['labels'] = y_val
        self.placeholders['labels_mask'] = val_mask
        self._loss()
        self._accuracy()
        return self.loss, self.accuracy

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
                                            back_act=lambda x: np.where(x == np.inf, 0, 1),
                                            dropout=True,
                                            sparse_inputs=False))
