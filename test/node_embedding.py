import numpy as np
import scipy.sparse as sp
from original.weights_preprocess import init_Weight
from original.optimizer import Adam
from original.layers import forward_hidden, backward_hidden
from original.metrics import forward_cross_entrocpy_loss, backward_cross_entrocpy_loss

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

# if __name__ == '__main__':
def load_karate(path="../_data/karate-club/"):
    """Load karate club dataset:  features: (34, 34)  adj: (34, 34), labels: (34, 4), edges: (78, 2)"""
    # path="../_data/karate-club/"
    print('Loading karate club dataset...')

    edges = np.loadtxt("{}edges.txt".format(path), dtype=np.int32) - 1  # 0-based indexing
    features = sp.eye(np.max(edges+1), dtype=np.float32).tocsr()
    idx_labels = np.loadtxt("{}mod-based-clusters.txt".format(path), dtype=np.int32)
    idx_labels = idx_labels[idx_labels[:, 0].argsort()]
    labels = encode_onehot(idx_labels[:, 1])

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    return adj.todense(), features.todense(), labels, edges

class GCN_Embedding:
    """GCN use for embedding"""
    def __init__(self, load_data_function, data_str="cora", hidden_unit=4, output_dim=2, learning_rate=0.01, weight_decay=5e-4):
        self.adj, self.features, self.labels, self.edges = load_data_function(0)
        # init
        self.n, self.f, self.c = self.adj.shape[0], self.features.shape[1], output_dim
        self.h = hidden_unit
        # init weight
        self.weight_inputs = init_Weight((self.f, self.h))
        self.weight_hidden = init_Weight((self.h, self.h))
        self.weight_outputs = init_Weight((self.h, self.c))

        self.adam_weight_inputs = Adam(weights=self.weight_inputs, learning_rate=learning_rate)
        self.adam_weight_hidden = Adam(weights=self.weight_hidden, learning_rate=learning_rate)
        self.adam_weight_outputs = Adam(weights=self.weight_outputs, learning_rate=learning_rate)

        self.layer_inputs = np.zeros((self.n, self.h))
        self.layer_hidden = np.zeros((self.n, self.h))
        self.layer_outputs = np.zeros((self.n, self.c))

        self.weight_decay = weight_decay

        # test
        self.grad_loss = None
        self.grad_weight_outputs = None
        self.grad_hidden = None
        self.grad_weight_hidden = None

    def one_train(self):
        self.layer_inputs = forward_hidden(self.adj, self.features, self.weight_inputs, act=lambda x: np.maximum(x, 0))  # act is tanh
        self.layer_hidden = forward_hidden(self.adj, self.layer_inputs, self.weight_hidden, act=lambda x: np.maximum(x, 0))
        self.layer_outputs = forward_hidden(self.adj, self.layer_hidden, self.grad_weight_outputs, act=lambda x: np.maximum(x, 0))

        # loss is not clearly, so give up
        pass

    def one_update(self):
        pass

#     early_stopping = 10; ephochs = 300; data_str = "pubmed"; dropout = 0.5; hidden_unit = 16; learning_rate = 0.01; weight_decay = 5e-4   # paras provided

