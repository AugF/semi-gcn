from networkx.algorithms.community import greedy_modularity_communities
import numpy as np

def load_karate_data():
    import networkx as nx
    G = nx.karate_club_graph()
    colors = list(greedy_modularity_communities(G))

    n = G.__len__()
    c = len(colors)

    A = np.zeros((n, n))
    for e in G.edges():
        A[e[0]][e[1]] = 1
        A[e[1]][e[0]] = 1

    Y = np.zeros((n, c))

    for i in range(n):  # 标号
        for j, color in enumerate(colors):
            if i in color:
                Y[i][j] = 1
    X = np.identity(n)
    return A, X, Y, colors, G

def picture(G, colors, pos):
    import networkx as nx
    import matplotlib.pyplot as plt
    nx.draw_networkx_nodes(G, pos, nodelist=colors[0], node_color='r', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=colors[1], node_color='b', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=colors[2], node_color='g', node_size=100)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges)
    plt.show()
    return

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str):
    import pickle as pkl
    import sys
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    return x, y, tx, ty, allx, ally, graph   # graph 邻接矩阵

if __name__ == '__main__':
    picture(100)