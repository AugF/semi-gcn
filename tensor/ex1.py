import tensorflow as tf

import numpy as np

np.random.seed(1)

inputs = np.random.random((4, 3))
weights = np.random.random((3, 1))
targets = np.array([1., 1., 0., 1.]).reshape(-1, 1)

np_dot = np.dot(inputs, weights)
# np_sigmoid = 0.5 * (np.tanh(np_dot) + 1)
np_sigmoid = 1 / (1 + np.exp(-np_dot)) # y = 1 / (1 + exp(-x))
np_labels = np_sigmoid * targets + (1 - np_sigmoid) * (1 - targets)
np_log = np.log(np_labels)
np_loss = - np.sum(np_log)

print("np dot", np_dot)
print("np sigmoid", np_sigmoid)
print("np targets", targets)
print("np labels", np_labels)
print("np log", np_log)
print("np loss", np_loss)

sess = tf.Session()
tf_inputs = tf.constant(inputs)
tf_weights = tf.constant(weights)
tf_targets = tf.constant(targets)

tf_dot = tf.matmul(tf_inputs, tf_weights)
tf_sigmoid = tf.sigmoid(tf_dot)
tf_labels = tf_sigmoid * targets + (1 - tf_sigmoid) * (1 - targets)
tf_log = tf.log(tf_labels)
tf_loss = - tf.reduce_sum(tf_log)

res = sess.run([tf_dot, tf_sigmoid, tf_targets, tf_labels, tf_log, tf_loss])

print("tf dot", res[0])
print("tf sigmoid", res[1])
print("tf targets", res[2])
print("tf labels", res[3])
print("tf log", res[4])
print("tf loss", res[5])