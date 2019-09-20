"""adam optimizer"""
import autograd.numpy as np  # autograd must use numpy
import tensorflow as tf
from autograd import grad

class Adam:
    """adam optimizer"""
    def __init__(self, weights, grad_function, learning_rate=0.001):
        """params, init"""
        self.learning_rate = learning_rate
        self.theta_t = weights
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.m_t = np.zeros(weights.shape)
        self.v_t = np.zeros(weights.shape)
        self.t = 0
        self.grad_function = grad_function

    def minimize(self):
        """more efficient"""
        self.t += 1
        g_t = self.grad_function(self.theta_t)
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * g_t * g_t
        alpha_t = self.learning_rate * ((1 - self.beta_2 ** self.t) ** 0.5) / (1 - self.beta_1 ** self.t)
        self.theta_t -= alpha_t * self.m_t / (self.v_t ** 0.5 + self.epsilon)


# build a toy dataset: numbers=4
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])
weights = np.array([0.0, 0.0, 0.0])

# define loss
def sigmoid(x):
    return 0.5*(np.tanh(x) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

# check loss
def numerical_loss(funciton, weights):
    """weigths: 1 dimension array"""
    h = 1e-8
    numerical_grad = np.zeros(weights.shape)

    for i in range(weights.shape[0]):
            weights[i] += h
            loss1 = training_loss(weights)
            weights[i] -= 2*h
            loss2 = training_loss(weights)
            numerical_grad[i] = (loss1 - loss2) / (2*h)
            weights[i] += h

    return numerical_grad

def test_autograd():
    """test whether grad in autograd is true"""
    print("numerical_grad", numerical_loss(training_loss, weights))
    training_gradient_fun = grad(training_loss)
    gradient = training_gradient_fun(weights)
    print("gradient", gradient)

# sgd
def sgd(epochs=1000):
    """stochastic according to inputs in loss function.inputs not need to be entire dataset."""
    grad_function = grad(training_loss)
    weights = np.array([0., 0., 0.])
    for i in range(epochs):
        weights -= grad_function(weights) * 0.01

    print("Trained loss: {}".format(training_loss(weights)))
    print("weights: {}".format(weights))

def test_adam(epochs=1000):
    weights = np.array([0., 0., 0.])
    adam = Adam(weights=weights, grad_function=grad(training_loss), learning_rate=0.01)
    for i in range(epochs):
        adam.minimize()
    print("Trained loss: {}".format(training_loss(adam.theta_t)))
    print("weights: {}".format(adam.theta_t))

# tensorflow
# tf_var = tf.get_variable("var", initializer=tf.constant(theta_t))
# loss = tf.math.reduce_mean(tf_var)
# optimizer = tf.train.AdamOptimizer()
# opt_op = optimizer.compute_gradients(loss, tf_var)
#
# sess = tf.Session()
#
# for _ in range(10):
#     res = sess.run(opt_op)
#     print("res", res)

if __name__ == '__main__':
    test_adam()
