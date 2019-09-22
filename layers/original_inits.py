import numpy as np

def init_Weight(shape):
    """Glorot & Bengio (AISTATS 2010) init"""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return initial

def init_dropout(shape, dropout):
    """Dropout 2014, * input"""
    col = np.array([1] * shape[0]).reshape(-1, 1)
    mat = np.repeat(col, shape[1], axis=1)
    return np.random.binomial(mat, dropout)