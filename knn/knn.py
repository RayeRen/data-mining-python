import numpy as np


def knn(X, X_train, y_train, K):
    """
    KNN k-Nearest Neighbors Algorithm.
    :param X: testing sample features, P-by-N_test matrix.
    :param X_train: training sample features, P-by-N matrix.
    :param y_train: training sample labels, 1-by-N row vector.
    :param K: the k in k-Nearest Neighbors
    :return: predicted labels, 1-by-N_test row vector.
    """
    n_pred = X.shape[1]
    n_train = X_train.shape[1]
    dist = np.linalg.norm(np.expand_dims(X, 2) - np.expand_dims(X_train, 1), axis=0)
    assert dist.shape == (n_pred, n_train)
    dist_argsort = np.argsort(dist, axis=1)
    labels = np.take(y_train[0], dist_argsort[:, :K])
    labels = labels.astype('int64')
    label_pred = np.array([np.bincount(labels[i]).argmax() for i in range(n_pred)]).reshape(1, -1)
    assert label_pred.shape == (1, n_pred)
    return label_pred
