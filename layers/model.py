from layers.utils import load_data
from layers.inits import preprocess_adj, preprocess_features, preprocess_Delta
from layers.inits import init_dropout, init_Weight

# 1. prepare data

# step 1. get original data
# adj--A,  features--X, labels--Y , train_mask, val_mask, test_mask
adj, features, labels, train_mask, val_mask, test_mask = load_data("cora")

# step 2. get preprocessed data

delta = preprocess_Delta(adj)

# a. get symmetrically normalize adj matrix  (position_list, value_list, shape)
adj = preprocess_adj(adj)   # every layer is the same
# b. row normalize features,  same to adj
features = preprocess_features(features)

# 2. intialize model

# W0,  W1
class Model(object):
    def __init__(self, input_size, hidden_size, outputsize):
        """init feathers, labels, input_dim"""
        pass

    def _loss(self):
        """return loss, grad"""
        pass

    def _accuracy(self):
        """return accuracy"""
        pass

    def _build(self):
        """add layers"""
        pass

class Layer(object):
    def __init__(self):
        """something special: w1, sparse_flag,"""

    def _call(self):
        """return output? no layers"""
        # tensorflow just use numerical ways, so grad only needs _call. don't need backward.


# 3. train


# train_loss, train_accuracy = model.fit(x_train, y_train)  update(w0, w1)
# val_loss, val_accuracy = model.evaluate(x_val, y_val)
# for epochs,  stop

# 4. test
# one time:  test_loss, test_accuracy = model.evalute(x_val, y_val)

