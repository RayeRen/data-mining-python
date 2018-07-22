import numpy as np


def kmeans(X, K):
    """
    KMEANS K-Means clustering algorithm
    :param X: data point features, n-by-p maxtirx.
    :param K: the number of clusters
    :return:
    idx  - cluster label
    ctrs - cluster centers, K-by-p matrix.
    iter_ctrs - cluster centers of each iteration, K-by-p-by-iter 3D matrix.
    """
    p = X.shape[1]
    iter_ctrs = []
    ctrs = np.random.uniform(0, 255, (K, p))
    iter_ctrs.append(ctrs)
    limit = 100
    while limit > 0:
        dist = np.linalg.norm(np.expand_dims(X, 1) - np.expand_dims(ctrs, 0), axis=-1)
        idx = dist.argmin(axis=1)
        ctrs_t = np.array([X[idx == k, :].mean(axis=0) for k in range(K)])
        ctrs_t[np.isnan(ctrs_t)] = ctrs[np.isnan(ctrs_t)]
        ctrs = ctrs_t
        iter_ctrs.append(ctrs)
        cdist = np.linalg.norm(ctrs - iter_ctrs[-2]) / K
        if cdist < 0.1:
            break
        limit -= 1
    iter_ctrs = np.array(iter_ctrs)
    return idx, ctrs, iter_ctrs
