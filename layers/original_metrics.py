import numpy as np
from layers.original_utils import softmax

# cross-entrocpy loss
def forward_cross_entrocpy_loss(outputs, y_onehot, mask):
    """y_onehot: one_hot. train_mask: []"""
    softmax_x = softmax(outputs)
    cross_sum = -np.multiply(y_onehot, np.log(softmax_x))
    cross_sum = (lambda x: x.reshape(x.shape[0],))(np.sum(cross_sum, axis=1)).astype(np.float32)
    # start operation
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    cross_sum = np.multiply(cross_sum, mask)
    return np.mean(cross_sum)

def backward_cross_entrocpy_loss(outputs, y_onehot, train_mask):
    """require shape: outputs.shape"""
    dX = softmax(outputs) - y_onehot
    dX = np.multiply(dX, train_mask.reshape(-1, 1))
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
    correct_predictions = np.equal(np.argmax(preds, axis=1), np.argmax(labels, axis=1))
    accuracy_all = np.array(correct_predictions, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)