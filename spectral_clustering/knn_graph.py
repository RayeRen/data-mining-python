import numpy as np


def knn_graph(X, k, threshold):
    """
    KNN_GRAPH Construct W using KNN graph
    :param X: data point features, n-by-p maxtirx.
    :param k: number of nn.
    :param threshold: distance threshold.
    :return: adjacency matrix, n-by-n matrix.
    """
    dist = np.linalg.norm(np.expand_dims(X, 0) - np.expand_dims(X, 1), axis=-1)
    dist_argsort = np.argsort(dist, axis=-1)
    mask = np.zeros_like(dist).astype(bool)
    for idx, c in enumerate(dist_argsort):
        mask[idx, c[1:k + 1]] = True
    mask &= dist < threshold
    dist = np.exp(-dist ** 2)
    dist[~mask] = 0
    dist = np.maximum(dist.T, dist)
    return dist
