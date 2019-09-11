import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def get_res_sum_j(T):
    T_sum = np.sum(T, axis=1)
    # res = T / np.tile(T_sum,(2,1)).T
    res = np.dot(np.diag(T_sum), T)  # 左乘
    return res


def get_res_sum_i(T):
    T_sum = np.sum(T, axis=0)
    # res = T / np.tile(T_sum,(2,1))
    # res = T / T_sum
    res = np.dot(T, np.diag(T_sum))
    return res


def get_A_hat(A):
    assert isinstance(A, np.ndarray)
    A_tilde = A + np.eye(n)
    D_t = np.diag(np.sum(A_tilde, axis=1)) ** -0.5
    D = np.where(D_t == np.inf, 0, D_t)
    return D * A_tilde * D


def get_softmax(T):
    assert isinstance(T, np.ndarray)
    exp_min_max = lambda x: np.exp(x - np.max(x))
    denom = lambda x: 1.0 / np.sum(x)
    T = np.apply_along_axis(exp_min_max, 1, T)
    denominator = np.apply_along_axis(denom, 1, T)
    if len(denominator.shape) == 1:
        denominator = denominator.reshape((denominator.shape[0], 1))
    return T * denominator


def get_ReLU(T):
    assert isinstance(T, np.ndarray)
    return np.where(T < 0, 0, T)


def main(A, X, Y, Y_test):  # k表示隐藏层
    assert X.shape == (n, f) and A.shape == (n, n) and Y.shape == (n, c)

    A_hat = get_A_hat(A)

    X_f = np.zeros((n, f))
    H = np.zeros((3 * k + 1, n, h))
    Z = np.zeros((n, c))

    def update_weights():
        # 梯度下降
        global W_f
        global W_h
        global W_c

        dL = (Z - 1) * Y
        W_c = W_c - elta * np.dot(H[3 * k].T, dL)
        dh = np.dot(A_hat.T, np.dot(dL, W_c.T)) * np.where(H[3 * k - 1] > 0, 1, 0)  # 中间值

        for i in range(1, k):
            T = elta * np.dot(H[3 * k - 3 * i].T, dh)
            dh = np.dot(A_hat.T, np.dot(dh, W_h[k - i - 1].T)) * np.where(H[3 * k - 3 * i - 1] > 0, 1, 0)
            W_h[k - i - 1] = W_h[k - i - 1] - T

        if (k > 1):
            dh = np.dot(A_hat.T, np.dot(dh, W_h[0].T)) * np.where(H[2] > 0, 1, 0)
        W_f = W_f - elta * np.dot(X_f.T, dh)

        return

    preloss = 0
    count = 0

    for loop in range(200):
        # one iteration
        for i in range(k + 1):
            if (i == 0):
                X_f = np.dot(A_hat, X)
                H[1] = np.dot(X_f, W_f)
                # H[2] = get_ReLU(H[1])  # ReLU函数
                H[2] = np.tanh(H[1])
            elif (i == k):
                H[3 * i] = np.dot(A_hat, H[3 * i - 1])
                Z = get_softmax(np.dot(H[3 * i], W_c))  # softmax函数
                # Z = np.tanh(np.dot(H[3*i], W_c))
            else:
                H[3 * i] = np.dot(A_hat, H[3 * i - 1])
                H[3 * i + 1] = np.dot(H[3 * i], W_h[i - 1])
                # H[3 * i + 2] = get_ReLU(H[3 * i + 1])
                H[3*i+2] = np.tanh(H[3*i+1])

        # Z_tp = np.log(Z)
        # L = np.where(Z_tp == -np.inf, 0, Z_tp) * (-Y)
        L = 0.5 * np.sum((np.argmax(Y) - np.argmax(Z)) ** 2)

        # 当数特别大时，会出现0异常， 这里用到了softmax函数的性质 softmax(x)=softmax(x+c), c=-np.max(a,axis=1)
        loss = np.sum(L)
        print("iteration {}, loss: {}".format(loop, loss))
        if (preloss >= loss):
            count += 1
            preloss = loss
        else:
            count = 0
        if (count == 10):  break
        update_weights()

    return


if __name__ == '__main__':
    import networkx as nx

    G = nx.karate_club_graph()

    from networkx.algorithms.community import greedy_modularity_communities

    colors = list(greedy_modularity_communities(G))

    # 超参数
    n = G.__len__()
    f = n
    h = 4
    c = len(colors)
    k = 1
    elta = 0.05

    A = np.zeros((n, n))
    for e in G.edges():
        A[e[0]][e[1]] = 1
        A[e[1]][e[0]] = 1

    u = 28
    Y = np.zeros((n, c))
    for i in range(n):
        for j, color in enumerate(colors):
            if i in color:
                    Y[i][j] = 1

    X = np.random.random((n, f))
    W_f = np.random.random((f, h))
    W_h = np.random.random((k - 1, h, h))  # 注意这里n怎么用？
    W_c = np.random.random((h, c))

    # print(colors)
    main(A, X, Y, Y_test)


