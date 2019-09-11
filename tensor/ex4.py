"""A very simple MNIST classifier.

See extensive documentation at

http://tensorflow.org/tutorials/mnist/beginners/index.md

"""

from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

# Import data

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')  # 把数据放在/tmp/data文件夹中

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)  # 读取数据集

# 建立抽象模型

x = tf.placeholder(tf.float32, [None, 784])  # 占位符

y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

a = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数和训练方法

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))  # 损失函数为交叉熵

optimizer = tf.train.GradientDescentOptimizer(0.5)  # 梯度下降法，学习速率为0.5

train = optimizer.minimize(cross_entropy)  # 训练目标：最小化损失函数

# Test trained model

correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Train

sess = tf.InteractiveSession()  # 建立交互式会话

tf.initialize_all_variables().run()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    train.run({x: batch_xs, y: batch_ys})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))