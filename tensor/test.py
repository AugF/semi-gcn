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

if __name__ == '__main__':
    test_accuracy()