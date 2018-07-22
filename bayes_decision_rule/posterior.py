import numpy as np

from likelihood import likelihood


def posterior(x):
    l = likelihood(x)
    total = np.sum(x)
    p_w = np.sum(x, axis=1, keepdims=True) / total
    pos = l * p_w
    px = np.sum(x, axis=0) / total
    return pos / px
