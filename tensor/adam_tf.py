import tensorflow as tf
import numpy as np
from tensor.adam_ex1 import test_adam

inputs = np.array([[0.52, 1.12, 0.77],
                           [0.88, -1.08, 0.15],
                           [0.52, 0.06, -1.30],
                           [0.74, -2.49, 1.39]])
targets = np.array([1., 1., 0., 1.]).reshape(-1, 1)  # tensorflow不支持类型推断, 即不同类型相乘
weights = np.array([0., 0., 0.]).reshape(-1, 1)

weights = tf.compat.v1.get_variable(name="vvvv", initializer=weights)
inputs = tf.constant(inputs)
targets = tf.constant(targets)

# get loss
dot1 = tf.matmul(inputs, weights)
preds = tf.sigmoid(dot1)
labels_pro = preds * targets + (1 - preds) * (1 - targets)
log_pro = tf.log(labels_pro)
loss = - tf.reduce_sum(log_pro)

# add adam
adam = tf.train.AdamOptimizer(0.01)
opt_op = adam.minimize(loss)

# train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("tensorflow: ")

for i in range(20):
    outs = sess.run([opt_op, loss, weights, preds, labels_pro, log_pro])
    print("iteration ", i)
    print("preds", outs[3])
    print("labes_pro", outs[4])
    print("log_pro", outs[5])
    print("loss", outs[1])
    print("weights", outs[2])
    break

print("adam: ")
test_adam(20)