import tensorflow as tf
import numpy as np

def wrapper(x):
    """wrapper for print"""
    return x.reshape(x.shape[0])


class Adam:
    """adam optimizer"""
    def __init__(self, weights, learning_rate=0.001):
        """params, init"""
        self.learning_rate = learning_rate
        self.theta_t = weights
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.m_t = np.zeros(weights.shape)
        self.v_t = np.zeros(weights.shape)
        self.t = 0
        # self.grad_function = grad_function

    def minimize(self, g_t):
        """more efficient"""
        self.t += 1
        # g_t = self.grad_function(self.theta_t)
        alpha_t = self.learning_rate * ((1 - self.beta_2 ** self.t) ** 0.5) / (1 - self.beta_1 ** self.t)
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * g_t * g_t
        self.theta_t -= alpha_t * self.m_t / (self.v_t ** 0.5 + self.epsilon)


def test_tensorflow(step=1, epochs=1000):
    inputs = np.array([[0.52, 1.12, 0.77],
                       [0.88, -1.08, 0.15],
                       [0.52, 0.06, -1.30],
                       [0.74, -2.49, 1.39]])
    targets = np.array([1., 1., 0., 1.]).reshape(-1, 1)  # tensorflow不支持类型推断, 即不同类型相乘
    weights = np.array([0., 0., 0.]).reshape(-1, 1)

    weights = tf.compat.v1.get_variable(name="v", initializer=weights)
    inputs = tf.constant(inputs)
    targets = tf.constant(targets)

    # get loss
    dot1 = tf.matmul(inputs, weights)
    preds = tf.sigmoid(dot1)
    labels_pro = preds * targets + (1 - preds) * (1 - targets)
    log_pro = tf.log(labels_pro)
    loss = - tf.reduce_sum(log_pro)

    # add adam
    adam = tf.train.AdamOptimizer(0.01)
    opt_op = adam.minimize(loss)

    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("tensorflow train: ")

    for i in range(epochs):
        outs = sess.run([opt_op, loss, weights])
        if i % step == 0:
            print("iteration: {}, loss: {}, weights: {}".format(i, outs[1], wrapper(outs[2])))


def test_adam(step=1, epochs=1000):
    inputs = np.array([[0.52, 1.12, 0.77],
                       [0.88, -1.08, 0.15],
                       [0.52, 0.06, -1.30],
                       [0.74, -2.49, 1.39]])
    targets = np.array([1., 1., 0., 1.]).reshape(-1, 1)

    weights = np.array([0., 0., 0.]).reshape(-1, 1)

    print("adam train")
    adam = Adam(weights=weights, learning_rate=0.01)
    for i in range(epochs):
        weights = adam.theta_t
        np_dot = np.dot(inputs, weights)
        np_sigmoid = 1 / (1 + np.exp(-np_dot))  # y = 1 / (1 + exp(-x))
        np_labels = np_sigmoid * targets + (1 - np_sigmoid) * (1 - targets)
        np_log = np.log(np_labels)
        np_loss = - np.sum(np_log)

        dloss = np.where(np.zeros(np_log.shape) == 0, -1, -1)
        dlog = 1 / np_labels
        dlabels = 2 * targets - 1
        dsigmoid = np_sigmoid - np_sigmoid ** 2  # todo 符号不对！ why?
        ddot = inputs.T  # dot
        g_t = np.dot(ddot, dsigmoid * dlabels * dlog * dloss)

        adam.minimize(g_t)

        print("iteration: {}, loss: {}, weights: {}".format(i, np_loss, wrapper(adam.theta_t)))

if __name__ == '__main__':
    test_tensorflow(epochs=5)
    test_adam(epochs=5)