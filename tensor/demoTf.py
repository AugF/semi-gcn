import tensorflow as tf
import numpy as np
from gcn.utils import onehot

n, c = 4, 3
y = np.random.randint(0, c, (n,))
y_logits = onehot(y, c)

# 1. prepare data
tf_hidden = tf.constant(np.random.random((n, c)))
tf_y_real = tf.constant(y_logits)

# 2. softmax
# way 1: softmax + reduce_sum
tf_softmax_x = tf.nn.softmax(tf_hidden)
tf_cross_entropy = - tf.reduce_sum(tf_y_real * tf.log(tf_softmax_x))

# way 2: softmax_cross_entropy
tf_cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=tf_hidden, labels=tf_y_real)
tf_cross_entropy_1 = tf.reduce_sum(tf_cross_entropy_logits)

with tf.Session() as sess:
    # run， at most  feed_dict()
    res,  res_logits, res_1 = sess.run([tf_cross_entropy, tf_cross_entropy_logits, tf_cross_entropy_1])
    print("res", res)
    print("res logits", res_logits)
    print("res 1", res_1)
