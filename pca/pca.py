import numpy as np


def pca(data):
    """

    :param data: Data matrix. Each row vector of fea is a data point.
    :return: eigvector - Each column is an embedding function, for a new
                           data point (row vector) x,  y = x*eigvector
                           will be the embedding result of x.
            eigvalue  - The sorted eigvalue of PCA eigen-problem.
    """
    data = data - np.mean(data, axis=1, keepdims=True)
    threshold = 0.01
    c = np.cov(data)
    eigvalue, eigvector = np.linalg.eig(c)
    eigvalue_argsort = np.argsort(eigvalue)[::-1]
    eigvalue = eigvalue[eigvalue_argsort]
    eigvector = eigvector[:, eigvalue_argsort]
    thre_index = np.abs(eigvalue) > threshold
    eigvalue = eigvalue[thre_index]
    eigvector = eigvector[:, thre_index]
    return eigvector.astype(float), eigvalue.astype(float)
