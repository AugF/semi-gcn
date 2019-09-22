from layers.hiddenLayer import forward_hidden, backward_hidden
from gcn.utils import prepare_gcn, numerical_grad
from gcn.utils import load_data
from gcn.inits import preprocess_adj, preprocess_features, preprocess_Delta
from gcn.inits import init_Weight, masked_accuracy
from layers.softmaxloss import softmax, _loss, backward
from gcn.adam import Adam


def gcn():
    A, P, X, W0, W1, Y, train_mask = prepare_gcn()

    # 1. forward

    # H0=relu(AXW0)
    H0, H0_tilde, H0_hat = forward_hidden(X, A, W0)  # n, h

    # H1=relu(AH0 W1)
    H1, H1_tilde, H1_hat = forward_hidden(H0, A, W1)

    # loss
    loss = _loss(H1, Y, P, train_mask)

    # 2. backward
    dH1 = backward(H1, Y, P, train_mask)  # n, c

    dW1, dH0 = backward_hidden(A, W1, dH1, H1_tilde, H1_hat)

    dW0, _ = backward_hidden(A, W0, dH0, H0_tilde, H0_hat)
    print(dW0)

    # 3. check grad
    f_w1 = lambda w1: _loss(forward_hidden(forward_hidden(X, A, W0)[0], A, w1)[0], Y, P, train_mask)
    f_w0 = lambda w0: _loss(forward_hidden(forward_hidden(X, A, w0)[0], A, W1)[0], Y, P, train_mask)

    grad_w1 = numerical_grad(f_w1, W1)
    grad_w0 = numerical_grad(f_w0, W0)
    print(grad_w0)


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


