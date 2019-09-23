import numpy as np
import tensorflow as tf
from layers.original_utils import onehot
from layers.original_layers import softmax

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
    np.random.seed(1)
    np_weight_init = np.random.uniform(low=-init_range, high=init_range, size=shape)
    sess = tf.Session()
    tf.random.set_random_seed(1)
    weight_init = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    tf_weight_init = sess.run(weight_init)
    print("np_weight_init", np_weight_init)
    print("tf_weight_init", tf_weight_init)


def test_soft_cross_entropy():
    outputs = np.random.random((4, 3))
    y = np.array([0, 1, 2, 0])
    y_onehot = onehot(y, 3)

    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_sum = np.sum(cross_sum, axis=1)
    np_loss = np.mean(cross_sum)

    tf_outputs = tf.constant(outputs)
    tf_y_onehot = tf.constant(y_onehot)
    tf_cross_sum = tf.nn.softmax_cross_entropy_with_logits(logits=tf_outputs, labels=tf_y_onehot)
    tf_loss = tf.reduce_mean(tf_cross_sum)
    sess = tf.Session()
    tf_outs = sess.run([tf_loss, tf_cross_sum])
    print("np_loss", np_loss)
    print("tf_loss", tf_outs[0])
    print("cross_sum", cross_sum)
    print("tf_cross_sum", tf_outs[1])

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking"""
    correct_predictions = np.equal(np.argmax(preds, axis=1), np.argmax(labels, axis=1))
    print("correct_pred", correct_predictions)
    accuracy_all = np.array(correct_predictions, dtype=np.float32)
    print("acc_all", accuracy_all)
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    print("mask", mask)
    accuracy_all *= mask
    print("acc_all after", accuracy_all)
    return np.mean(accuracy_all)

def test_accuracy():
    preds = np.array([[0.1, 0.6, 0.3],
                      [0.3, 0.5, 0.2],
                      [0.9, 0.1, 0.],
                      [0.5, 0.2, 0.3]])
    labels = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]])
    mask = np.array([0, 1, 1, 0])
    res = masked_accuracy(preds, labels, mask)
    print("res", res)

def test_write_log(log_str="test"):
    import os
    print(os.getcwd())
    log_file_path = "../log/{}.txt".format(log_str)
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as f:
            print(f)

    with open(log_file_path, "a") as f:
        for i in range(10):
            f.write(str(i) + "\n")

def testfun(x=1):
    return x

def test_relu():
    x = np.random.random((2, 2))
    with tf.Session() as sess:
        tf_relu = sess.run(tf.nn.relu(x))
        np_relu = np.maximum(x, 0)
        print("tf_relu", tf_relu)
        print("np_relu", np_relu)


def print_seed():
    print(np.random.get_state()[1][0])


def test_writeAndread():
    m, n = 2, 2
    log_file_path = "../log/test.txt"
    res = np.random.random((m, n))
    with open(log_file_path, "w") as f:
        for i in range(m):
            for j in range(n):
                f.write(str(res[i, j]) + " ")
            f.write("\n")

    ans = np.zeros((m, n), dtype=np.float)
    with open(log_file_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            for j, v in enumerate(line.strip().split(" ")):
                ans[i, j] = float(v)
    print("res", res)
    print("ans", ans)
    print("res == ans", (res == ans).all())




