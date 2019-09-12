import numpy as np
import matplotlib.pyplot as plt
from layers.utils import onehot

def picture(pos, edges, labs):
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
    plt.show()

if __name__ == '__main__':
    # sample test
    n, c = 34, 4
    labs = np.random.randint(c, size=(n, ))
    labs = onehot(labs, c)

    pos = np.random.uniform(-1, 1, size=(n, 2))
    edges = set()
    for i in range(50):
        edge = tuple(np.sort(np.random.choice(n, (2,))))
        edges.add(edge)
    picture(pos, edges, labs)
