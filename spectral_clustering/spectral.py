import numpy as np
from sklearn.cluster import KMeans


def spectral(W, k):
    """
    SPECTRUAL spectral clustering
    :param W: Adjacency matrix, N-by-N matrix
    :param k: number of clusters
    :return: data point cluster labels, n-by-1 vector.
    """
    w_sum = np.array(W.sum(axis=1)).reshape(-1)
    D = np.diag(w_sum)
    _D = np.diag(w_sum ** (-1 / 2))
    L = D - W
    L = _D @ L @ _D
    eigval, eigvec = np.linalg.eig(L)
    eigval_argsort = eigval.argsort()
    F = np.take(eigvec, eigval_argsort[:k], axis=-1)
    idx = KMeans(n_clusters=k).fit(F).labels_
    return idx
