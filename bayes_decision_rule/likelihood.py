import numpy as np


def likelihood(x):
    c, n = x.shape
    l = x.copy()
    l /= np.sum(x, axis=1, keepdims=True)
    return l
