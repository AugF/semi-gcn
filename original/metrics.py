import tensorflow as tf
import numpy as np
from original.utils import softmax

def wrapper(x):
    return x.reshape(x.shape[0],)

# cross-entrocpy loss
def forward_cross_entrocpy_loss(outputs, y_onehot, mask):
    """y_onehot: one_hot. train_mask: []"""
    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_sum = wrapper(np.sum(cross_sum, axis=1)).astype(np.float32)  # todo, attention shape here!
    # start operation
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    cross_sum = np.multiply(cross_sum, mask)
    return np.mean(cross_sum)

def backward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """require shape: outputs.shape"""
    dX = softmax(outputs) - y_onehot
    # mask
    mask = np.array(train_mask, dtype=np.float32)
    mask /= np.mean(mask)
    dX = np.multiply(dX, mask.reshape(-1, 1))
    return dX / outputs.shape[0]


# l2 loss
def l2_loss(X):
    """for matrix, tf.nn.l2_loss: np.sum(x**2)/2"""
    x_square = X ** 2
    x_sum = np.sum(x_square)
    x_l2 = x_sum / 2
    return x_l2

# acc
def masked_accuracy(preds, labels, mask):
    """Accuracy with masking"""
    preds_max = np.argmax(preds, axis=1)
    labels_max = np.argmax(labels, axis=1)
    correct_predictions = np.equal(wrapper(preds_max), labels_max)
    accuracy_all = np.array(correct_predictions, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)


def tf_masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


if __name__ == '__main__':
    import numpy as np
    from original.utils import onehot, numerical_grad
    from original.load import sample_mask

    n, c = 2708, 7
    outputs = np.random.random((n, c))
    y_real = np.random.randint(low=0, high=c, size=(n, ))
    y_onehot = onehot(y_real, c)
    train_mask = sample_mask(range(500), (n, ))

    f = lambda x: forward_cross_entrocpy_loss(x, y_onehot, train_mask)
    np_grad = backward_cross_entrocpy_loss(outputs, y_onehot, train_mask)
    real_grad = numerical_grad(f, outputs)
    print("np_grad", np_grad)
    print("real_grad", real_grad)