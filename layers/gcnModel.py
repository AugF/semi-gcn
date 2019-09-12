import numpy as np
from layers.hiddenLayer import forward_hidden, backward_hidden
from layers.lossLayer import forward_loss, backward_grad
from layers.utils import prepare_gcn, numerical_grad

def gcn():
    A, P, X, W0, W1, Y, train_mask = prepare_gcn()

    # 1. forward

    # H0=relu(AXW0)
    H0, H0_tilde, H0_hat = forward_hidden(X, A, W0)  # n, h

    # H1=relu(AH0 W1)
    H1, H1_tilde, H1_hat = forward_hidden(H0, A, W1)

    # loss + reg loss
    loss = forward_loss(H1, Y, P, train_mask)

    # 2. backward

    dH1 = backward_grad(H1, Y, P, train_mask)  # n, c

    dW1, dH0 = backward_hidden(A, W1, dH1, H1_tilde, H1_hat)

    dW0, _ = backward_hidden(A, W0, dH0, H0_tilde, H0_hat)
    print(dW0)

    # 3. check grad
    f_w1 = lambda w1: forward_loss(forward_hidden(forward_hidden(X, A, W0)[0], A, w1)[0], Y, P, train_mask)
    f_w0 = lambda w0: forward_loss(forward_hidden(forward_hidden(X, A, w0)[0], A, W1)[0], Y, P, train_mask)

    # grad_w1 = numerical_grad(f_w1, W1)
    grad_w0 = numerical_grad(f_w0, W0)
    print(grad_w0)

if __name__ == '__main__':
    gcn()


