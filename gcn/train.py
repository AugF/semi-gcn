from gcn.utils import load_data, prepare_gcn
from gcn.inits import preprocess_adj, preprocess_features
from gcn.model import GCN

import time
import tensorflow as tf
import numpy as np

# Set random seed
seed = 123
np.random.seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 2, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('grad_step', 0.2, 'grad step')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# 1. prepare data
# step 1. get original data
# adj--A,  features--X, labels--Y , train_mask, val_mask, test_mask
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = prepare_gcn()

# step 2. get preprocessed data

# delta = preprocess_Delta(adj)

# a. get symmetrically normalize adj matrix  (position_list, value_list, shape)
adj = preprocess_adj(adj)   # every layer is the same
# b. row normalize features,  same to adj
features = preprocess_features(features)

# warp to placeholders
placeholders = {
    'features': features,
    'adj':  adj,
    # 'delta': delta,
    'dropout': 0.5,
    'labels_mask': train_mask,  # just for train
    'labels': y_train,          # onehot
    'num_features_nonzero': 0
}

# 2. intialize model

model = GCN(placeholders, input_dim=features.shape[1])

# 3. Train model
cost_val = []

for epoch in range(FLAGS.epochs):
    t = time.time()
    # Training step
    train_loss, train_acc = model.one_train()

    # Validation
    val_loss, val_acc = model.evaluate(y_val, val_mask)
    cost_val.append(val_loss)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.6f}".format(train_loss),
          "train_acc=", "{:.6f}".format(train_acc), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# 4. test
# one time:  test_loss, test_accuracy = model.evalute(x_val, y_val)
t = time.time()
test_cost, test_acc = model.evaluate(y_test, test_mask)
test_duration = time.time() - t

print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
