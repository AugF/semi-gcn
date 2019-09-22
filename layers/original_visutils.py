import matplotlib.pyplot as plt
import numpy as np


def draw_fig(cost_train, cost_val, acc_train, acc_val, fig_name):
    """use for drawing the fig"""
    max_val = max(max(cost_train), max(cost_val), max(acc_train), max(acc_val))
    min_val = min(min(cost_train), min(cost_val), min(acc_train), min(acc_val))

    x = np.arange(min_val, max_val + 1e-5, (max_val - min_val) / (len(cost_val) - 1))
    fig, ax = plt.subplots()

    ax.plot(x, cost_train, label="cost_train")
    ax.plot(x, cost_val, label='cost_val')
    ax.plot(x, acc_train, label='acc_train')
    ax.plot(x, acc_val, label='acc_val')
    legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    plt.savefig(fig_name)


def picture2D_outputs(pos, edges, labs, fig_name):
    """draw 2-D picture"""
    # labs: (n, c)    edges: [(a,b)]  pos: (n, 2)
    n, c = labs.shape

    # 1. get labs (n, 1)
    labs = np.argmax(labs, axis=1)

    # 2. get classes
    f = lambda x: np.argwhere(labs == x).ravel().tolist()

    # 3. draw point
    g = lambda li, idx: pos[li, ][:, idx]  #li, class: li,  idx:取x, y
    colors = ['r', 'g', 'b', 'c']
    if len(colors) < c:
        colors.extend(['c']*(c - len(colors)))

    for i, c in enumerate(colors):
        plt.scatter(g(f(i), 0), g(f(i), 1), color=c, s=50)

    # 4. draw edges
    for edge in edges:
        point1 = pos[edge[0]]
        point2 = pos[edge[1]]
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='k')

    # 5. save fig
    plt.imshow()
    # plt.imsave(fig_name)