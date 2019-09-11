import numpy as np

EPOCHS = 1000


class Adam:
    def __init__(self, weights, lr=0.01, beta1=0.9, beta2=0.999, epislon=1e-8):
        assert isinstance(weights, np.ndarray)
        self.theta = weights
        self.m = np.zeros(weights.shape)
        self.v = np.zeros(weights.shape)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon

    def minimize_once(self, gradient):
        assert isinstance(gradient, np.ndarray) and gradient.shape == self.theta.shape
        self.t += 1
        lr = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient * gradient)
        self.theta -= lr * self.m / (self.v ** 0.5 + self.epislon)
        return self.theta


class Model:
    def __init__(self, A, X, Y, h=2, l=0.85, lr=0.01, l2=5e-4):
        assert isinstance(A, np.ndarray) and isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)
        n, c = X.shape
        f = Y.shape[1]
        self.l = int(n * l)  # 有标记数据个数
        self.l2 = l2
        self.n = n
        self.c = c
        self.f = f
        self.h = h
        # 中间变量
        self.w0 = np.random.random((c, h))
        self.w1 = np.random.random((h, f))
        self.adm0 = Adam(self.w0, lr=lr)
        self.adm1 = Adam(self.w1, lr=lr)
        self.H = np.zeros((n, h))
        self.Z = np.zeros((n, f))
        # 特定值
        self.A = self.get_A(A)
        self.D = self.get_D(A)
        self.X = X
        self.Y = Y
        self.Y_test = np.copy(Y[self.l:, ])

    def get_D(self, A):
        D = np.diag(np.sum(A, axis=1))
        return D - A

    def get_A(self, A):
        assert isinstance(A, np.ndarray)
        A_tilde = A + np.eye(self.n)
        D = np.diag(np.sum(A_tilde, axis=1) ** 0.5)
        return D * A_tilde * D

    def get_softmax(self, T):
        # https://segmentfault.com/a/1190000010039529
        assert isinstance(T, np.ndarray)
        exp_min_max = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        T = np.apply_along_axis(exp_min_max, 1, T)
        denominator = np.apply_along_axis(denom, 1, T)
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))
        return T * denominator

    def _embeddingloss(self):
        t1 = np.dot(np.dot(self.A, self.X), self.w0)
        self.H = np.where(t1 < 0, 0, t1)
        t2 = np.dot(np.dot(self.A, self.H), self.w1)
        self.Z = self.get_softmax(t2)
        l = - self.Y * np.log(np.where(self.Z <= 0, 1, self.Z))
        return np.mean(l)

    def _update_embeddingweights(self):
        t = (self.Z - 1) * self.Y

        g0 = np.dot(self.A.T, np.dot(t, self.w1.T))
        g0 = np.where(self.H > 0, 1, 0) * g0
        a = np.dot(self.A, self.X)
        g0 = np.dot(a.T, g0)
        self.w0 = self.adm0.minimize_once(g0)
        # self.w0 -= 0.01*g0

        g1 = np.dot(self.A, self.H)
        g1 = np.dot(g1.T, t)
        self.w1 = self.adm1.minimize_once(g1) # adam 更新
        # self.w1 -= 0.01*g1
        return

    def _loss(self):
        t1 = np.dot(np.dot(self.A, self.X), self.w0)
        self.H = np.where(t1 < 0, 0, t1)
        t2 = np.dot(np.dot(self.A, self.H), self.w1)
        self.Z = self.get_softmax(t2)
        z = self.Z[:self.l, ]
        l = - self.Y[: self.l, ] * np.log(np.where(z <= 0, 1, z))
        loss = np.mean(l)

        loss += 2 * self.l2 * np.dot(np.dot(self.Z.T, self.D), self.Z)
        return loss

    def _predict(self):  # 预测问题
        res = np.argmax(self.Z[self.l:, ], axis=1) == np.argmax(self.Y_test, axis=1)
        return res, np.sum(res)

    def get_l0_g(self):  # labeled loss
        t = (self.Z - 1)[:self.l, ] * self.Y[:self.l, ]
        g0 = np.dot(self.A[:self.l, ].T, np.dot(t, self.w1.T))
        g0 = np.where(self.H > 0, 1, 0) * g0
        a = np.dot(self.A, self.X)
        g0 = np.dot(a.T, g0)
        g1 = np.dot(self.A, self.H)[:self.l, ]
        g1 = np.dot(g1.T, t)
        return g0, g1

    def get_l1_g(self):
        t = (self.Z - 1) * self.Y
        g0 = np.dot(self.A.T, np.dot(t, self.w1.T))
        g0 = np.where(self.H > 0, 1, 0) * g0
        a = np.dot(self.A, self.X)
        g0 = np.dot(a.T, g0)
        g1 = np.dot(self.A, self.H)
        g1 = np.dot(g1.T, t)
        return g0, g1

    def _update_weights(self):

        g00, g01 = self.get_l0_g()
        g10, g11 = self.get_l1_g()

        self.w0 -= 0.001* g00
        # g1 = np.dot(self.A, self.H)[:self.l, ]
        # g1 = np.dot(g1.T, t)

        self.w1 -= 0.001 * g01
        return

    def _build(self, epochs = 400):
        preloss = 0
        count = 0
        for i in range(epochs):
            loss = self._loss()
            # res, accuary = self._predict()
            self._update_weights()
            print("iteration {}, loss: {}".format(i, loss))
            if (preloss >= loss):
                count += 1
                preloss = loss
            else:
                count = 0
            if (count == 10):  break


def test_Adam():
    # https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E6%B3%95
    f = lambda x,y: x**2 + y**2
    w = np.array([3., 4.])

    def target_func(weights):
        x, y = weights
        return f(x, y)

    from autograd import grad
    adam = Adam(w, 0.01)
    for i in range(200):
        g = grad(target_func)
        loss = target_func(adam.theta)
        w = adam.minimize_once(g(adam.theta))
        print("w", w, "loss", loss)

def test_karate_data():
    from gcn import util
    A, X, Y, colors, G = util.load_karate_data()
    model = Model(A, X, Y)
    for i in range(300):
        # loss = model._embeddingloss()
        # model._update_embeddingweights()
        loss = model._loss()
        model._update_weights()
        if i % 100 == 0:
            pos = model.H
            print("begin drawing")
            util.picture(G, colors, pos)
        print("iterations", i, "loss", loss)

if __name__ == '__main__':
    test_karate_data()









