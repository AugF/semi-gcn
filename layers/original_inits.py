import numpy as np


def init_Weight(shape):
    """Glorot & Bengio (AISTATS 2010) init"""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return initial


def init_dropout(shape, dropout):
    """Dropout 2014, * input"""
    col = np.array([1] * shape[0]).reshape(-1, 1)
    mat = np.repeat(col, shape[1], axis=1)
    return np.random.binomial(mat, dropout)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking"""
    correct_predictions = np.equal(np.argmax(preds, axis=1), np.argmax(labels, axis=1))
    accuracy_all = np.array(correct_predictions, dtype=np.float32)
    mask = np.array(mask, dtype=np.float32)
    mask /= np.mean(mask)
    accuracy_all *= mask
    return np.mean(accuracy_all)


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