"""adam optimizer"""
import numpy as np
import math
import tensorflow as tf

np.random.seed(1)

n = 4
# data
theta_t = np.random.random((n, n)) # weight
x = np.random.random(n)
y_real = np.random.random(n)

b = np.random.random(n)
y = np.dot(theta_t, x) + b

square_loss = np.sum((y - y_real) ** 2)

grad = lambda x: 1 / x

# params
learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8   # to change

# init
m_t = np.zeros(n)
v_t = np.zeros(n)
t = 0

# algorithm

for _ in range(10):
    t = t + 1
    g_t = grad(theta_t)
    m_t = beta_1 * m_t + (1 - beta_1) * g_t
    v_t = beta_2 * v_t + (1 - beta_2) * np.power(g_t, 2)
    alpha_t = learning_rate * math.sqrt(1 - math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
    theta_t = theta_t - alpha_t *  m_t / (np.sqrt(v_t) + epsilon)
    print("theta_t", theta_t)

# tensorflow
tf_var = tf.get_variable("var", initializer=tf.constant(theta_t))
loss = tf.math.reduce_mean(tf_var)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
opt_op = optimizer.compute_gradients(loss, tf_var)

sess = tf.Session()

for _ in range(10):
    res = sess.run(opt_op)
    print("res", res)

