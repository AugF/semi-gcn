import numpy as np
import tensorflow as tf

def test_tf_reduce_mean():
    x = np.random.random((4, 5))
    y = np.mean(x)
    print("x", x)
    print("y", y)

    y1 = np.sum(x) / 20
    print("y1", y1)

    tf_x = tf.constant(x)
    tf_y = tf.reduce_mean(tf_x)

    sess = tf.Session()
    y_out = sess.run(tf_y)
    print("tf_y", y_out)


def test_l2_loss():
    x = np.random.random((4, 5))
    y = np.sum(x ** 2) / 2
    print("x", x)
    print("y", y)

    tf_x = tf.constant(x)
    tf_y = tf.nn.l2_loss(tf_x)

    sess = tf.Session()
    y_out = sess.run(tf_y)
    print("tf_y", y_out)


def test_random_uniform():
    """can't test"""
    shape = 4, 5
    init_range = np.sqrt(6.0 / (shape[0]*shape[1]))
    np_weight_init = np.random.uniform(low=-init_range, high=init_range, size=shape)
    sess = tf.Session()
    weight_init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    tf_weight_init = sess.run(weight_init)
    print("np", np_weight_init)
    print("tf", tf_weight_init)


if __name__ == '__main__':
    test_random_uniform()