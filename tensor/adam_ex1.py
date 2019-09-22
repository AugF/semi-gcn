"""adam optimizer"""
import autograd.numpy as grad_np  # autograd must use numpy
from autograd import grad
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


class LogisticModel:
    """logistic mode"""
    def __init__(self, epochs=1000):
        self.epochs = epochs
        self.inputs, self.targets = self.prepare_data()
        self.weights = np.zeros(self.inputs.shape[1], dtype=np.float)

    @staticmethod
    def prepare_data():
        """build a toy dataset"""
        inputs = np.array([[0.52, 1.12, 0.77],
                           [0.88, -1.08, 0.15],
                           [0.52, 0.06, -1.30],
                           [0.74, -2.49, 1.39]])
        targets = np.array([1., 1., 0., 1.]).reshape(-1, 1)
        return inputs, targets

    def training_loss(self, weights):
        # Training loss is the negative log-likelihood of the training labels.
        preds = self.logistic_predictions(weights)
        label_probabilities = preds * self.targets + (1 - preds) * (1 - self.targets)
        label_pro_log = grad_np.log(label_probabilities)
        return -grad_np.sum(label_pro_log)

    def logistic_predictions(self, weights):
        return self.sigmoid(grad_np.dot(self.inputs, weights))

    def sigmoid(self, x):
        return 1 / (1 + grad_np.exp(-x))

    def numerical_loss(self, function, weights):
        """tools, weights: 1 dimension array"""
        h = 1e-8
        numerical_grad = np.zeros(weights.shape)
        for i in range(weights.shape[0]):
            weights[i] += h
            loss1 = function(weights)
            weights[i] -= 2 * h
            loss2 = function(weights)
            numerical_grad[i] = (loss1 - loss2) / (2 * h)
            weights[i] += h
        return numerical_grad


def test_autograd():
    """test whether grad in autograd is true"""
    lgmodel = LogisticModel()
    numerical_loss, training_loss = lgmodel.numerical_loss, lgmodel.training_loss
    weights = np.array([0., 0., 0.])
    print("numerical_grad", numerical_loss(training_loss, weights))
    training_gradient_fun = grad(training_loss)
    gradient = training_gradient_fun(weights)
    print("gradient", gradient)


def test_sgd(epochs=1000, training_loss=LogisticModel().training_loss):
    """stochastic according to inputs in loss function.inputs not need to be entire dataset."""
    grad_function = grad(training_loss)
    weights = np.array([0., 0., 0.])
    for i in range(epochs):
        weights -= grad_function(weights) * 0.01

    print("Trained loss: {}".format(training_loss(weights)))
    print("weights: {}".format(weights))


def test_adam(step=1, epochs=1000):
    # lgmodel
    lgmodel = LogisticModel()
    inputs, targets, training_loss = lgmodel.inputs, lgmodel.targets, lgmodel.training_loss
    weights = np.array([0., 0., 0.]).reshape(-1, 1)
    # adam
    # g_t = np.dot(inputs.T, np.array([1]*inputs.shape[0]).reshape(-1, 1))
    # print("g_t", wrapper(g_t))
    g_t = lgmodel.numerical_loss(lambda x: np.sum(np.dot(inputs, x)), np.random.random((3, 1)))
    # print("check grad", wrapper(check_grad))

    print("adam train")
    adam = Adam(weights=weights, learning_rate=0.01)
    for i in range(epochs):
        weights = adam.theta_t.copy()
        adam.minimize(g_t)
        dot1 = np.dot(inputs, weights)
        loss = np.sum(dot1)
        if i % step == 0:
            print("iteration: {}, loss: {}, weights: {}".format(i, loss, wrapper(adam.theta_t)))


if __name__ == '__main__':
    test_adam(20)
