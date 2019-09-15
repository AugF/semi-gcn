import tensorflow as tf

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.007, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1600, 'Number of epochs to train.') # todo 200
flags.DEFINE_integer('hidden1', 4, 'Number of units in hidden layer 1.')  # todo chang back 16
flags.DEFINE_integer('hidden2', 2, 'Number of units in hidden layer 1.')  # todo chang back 16
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 2000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 5, 'Maximum Chebyshev polynomial degree.')