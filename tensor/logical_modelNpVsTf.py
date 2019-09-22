import tensorflow as tf

import numpy as np

np.random.seed(1)


def numerical_grad(g, weights):
    h = 1e-8
    check_grad = np.zeros(weights.shape)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            weights[i, j] += h
            loss1 = g(weights)
            weights[i, j] -= (2 * h)
            loss2 = g(weights)
            check_grad[i, j] = (loss1 - loss2) / (2 * h)
            weights[i, j] += h
    print("check_grad", check_grad)


inputs = np.random.random((4, 3))
weights = np.random.random((3, 1))
targets = np.array([1., 1., 0., 1.]).reshape(-1, 1)

np_dot = np.dot(inputs, weights)
# np_sigmoid = 0.5 * (np.tanh(np_dot) + 1)
np_sigmoid = 1 / (1 + np.exp(-np_dot)) # y = 1 / (1 + exp(-x))
np_labels = np_sigmoid * targets + (1 - np_sigmoid) * (1 - targets)
np_log = np.log(np_labels)
np_loss = - np.sum(np_log)

def print_np():
    print("np dot", np_dot)
    print("np sigmoid", np_sigmoid)
    print("np targets", targets)
    print("np labels", np_labels)
    print("np log", np_log)
    print("np loss", np_loss)


dloss = np.where(np.zeros(np_log.shape) == 0, -1, -1)
dlog = 1 / np_labels
dlabels = 2 * targets - 1
dsigmoid = np_sigmoid - np_sigmoid ** 2  # todo 符号不对！ why?
ddot = inputs.T  # dot

print("grad", np.dot(ddot, dsigmoid * dlabels * dlog * dloss))
fx = lambda x: 1 / (1 + np.exp(-np.dot(inputs, x)))
numerical_grad(lambda x: -np.sum(np.log(fx(x) * targets + (1 - fx(x)) * (1 - targets))), weights)


# tensorflow
def test_tensorflow():
    sess = tf.Session()
    tf_inputs = tf.constant(inputs)
    tf_weights = tf.constant(weights)
    tf_targets = tf.constant(targets)

    tf_dot = tf.matmul(tf_inputs, tf_weights)
    tf_sigmoid = tf.sigmoid(tf_dot)
    tf_labels = tf_sigmoid * targets + (1 - tf_sigmoid) * (1 - targets)
    tf_log = tf.log(tf_labels)
    tf_loss = - tf.reduce_sum(tf_log)

    res = sess.run([tf_dot, tf_sigmoid, tf_targets, tf_labels, tf_log, tf_loss])
    print("tf dot", res[0])
    print("tf sigmoid", res[1])
    print("tf targets", res[2])
    print("tf labels", res[3])
    print("tf log", res[4])
    print("tf loss", res[5])