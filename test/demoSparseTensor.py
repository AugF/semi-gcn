import tensorflow as tf

sess = tf.Session()
indices = [[0, 0], [1, 2]]
values = [1, 2]
dense_shape = [3, 4]

tf_sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
outs = sess.run(tf_sparse_tensor)  # SparseTensorValue

print(outs)

