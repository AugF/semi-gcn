# coding: utf-8
# 图的可视化

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# t1. import data
n, c = 34, 4
labels = np.random.randint(0,4, size=(n,))
pos = np.random.uniform(-1,1, size=(n,2))

# 2. draw point

g = lambda li, idx: pos[li, ][:, idx]
f = lambda x: np.argwhere(labels==x).ravel().tolist()

plt.scatter(g(f(0),0), g(f(0), 1), color='r', s=50)
plt.scatter(g(f(1),0), g(f(1), 1), color='g', s=50)
plt.scatter(g(f(2),0), g(f(2), 1), color='b', s=50)
plt.scatter(g(f(3),0), g(f(3), 1), color='c', s=50)

# 3. draw edges
edges = nx.karate_club_graph().edges
for edge in edges:
    point1 = pos[edge[0]]
    point2 = pos[edge[1]]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='k')

plt.show()