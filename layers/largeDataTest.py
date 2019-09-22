from gcn.utils import load_data
from gcn.inits import preprocess_adj, preprocess_features, preprocess_Delta
from gcn.inits import init_Weight, masked_accuracy
from layers.softmaxloss import softmax
from layers.hiddenLayer import forward_hidden, backward_hidden
from gcn.adam import Adam


# 1. prepare data
# step 1. get original data
# adj--A,  features--X, labels--Y , train_mask, val_mask, test_mask
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data("cora")

# step 2. get preprocessed data

delta = preprocess_Delta(adj)

# a. get symmetrically normalize adj matrix  (position_list, value_list, shape)
adj = preprocess_adj(adj)   # every layer is the same
# b. row normalize features,  same to adj
features = preprocess_features(features)

# 2. inits
n = len(adj)
h = 16

weight0 = init_Weight((features.shape[1], h))
weight1 = init_Weight((h, y_train.shape[1]))

# train

for epoch in range(800):
    print("epoch", epoch)
    # train forward
    H0, H0_tilde, H0_hat = forward_hidden(features, adj, weight0)  # n, h
    H1, H1_tilde, H1_hat = forward_hidden(H0, adj, weight1)
    Y_pred = softmax(H1)
    train_loss = _loss(Y_pred, y_train, train_mask)
    train_accuracy = masked_accuracy(Y_pred, y_train, train_mask)
    print("train_loss: {}, train_accuracy: {}".format(train_loss, train_accuracy))

    # val loss
    h1 = forward_hidden(forward_hidden(features, adj, weight0)[0], adj, weight1)[0]
    y_pred = softmax(h1)
    val_loss = _loss(y_pred, y_val, val_mask)
    val_accuracy = masked_accuracy(y_pred, y_val, val_mask)
    print("val_loss: {}, val_accuracy: {}".format(val_loss, val_accuracy))

    # backward
    dH1 = backward(H1, y_train, train_mask)  # n, c
    dW1, dH0 = backward_hidden(adj, weight1, dH1, H1_tilde, H1_hat)
    dW0, _ = backward_hidden(adj, weight0, dH0, H0_tilde, H0_hat)

    # update weight
    weight1 -= 0.9 * dW1
    weight0 -= 0.9 * dW0