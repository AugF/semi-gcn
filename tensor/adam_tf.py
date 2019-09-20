import tensorflow as tf
import numpy as np
from tensor.adam_ex1 import test_adam


def wrapper(x):
    """wrapper for print"""
    return x.reshape(x.shape[0])

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

    for i in range(1000):
        _, loss, dot1, preds, labels_pro, log_pro = sess.run([opt_op, loss, dot1, preds, labels_pro, log_pro])
        print(wrapper(dot1))
        print(wrapper(preds))
        print(wrapper(labels_pro))
        print(wrapper(log_pro))
        print(loss)
        break
        # if i % step == 0:
        #     print("iteration: {}, loss: {}, weights: {}".format(i, outs[1], outs[2].reshape((outs[2].shape[0]))))


if __name__ == '__main__':
    test_adam()
    test_tensorflow()