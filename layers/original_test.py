import tensorflow as tf

def tf_manual_grad():
    import numpy as np
    np.random.seed(123)
    n, c = 4, 2
    learning_rate = 0.8

    y_real = tf.constant(np.random.randint(0, 1, size=(n, 1)), dtype=tf.float32)
    inputs = tf.constant(np.random.randint(1, 10, size=(n, 1)), dtype=tf.float32)
    weights = tf.Variable(initial_value=np.random.random((n, n)), dtype=tf.float32)
    y_preds = tf.matmul(weights, inputs)

    loss = tf.nn.l2_loss(y_preds - y_real)

    grad = tf.gradients(loss, weights)
    update = tf.assign(weights, weights - learning_rate * grad[0])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run([grad, y_preds, weights]))  # anything connected to var must be updated
        for _ in range(5):
            # run something about var
            weights_val, grad_val, loss_val = sess.run([weights, grad, loss])
            print("variable is weight: {}, grad: {}, and the loss is {}"
                  .format(weights_val, grad_val, loss_val))
            # update _weights
            print(sess.run(update))  # update
            break