import numpy as np
import matplotlib.pyplot as plt


def kmeans_plot(X, idx, ctrs, iter_ctrs):
    """
    KMEANS K-Means clustering algorithm
    :param X: data point features, n-by-p matrix.
    :param idx: cluster label
    :param ctrs: cluster centers, K-by-p matrix.
    :param iter_ctrs: cluster centers of each iteration, K-by-p-by-iter 3D matrix.
    :return:
    """
    plt.figure()
    plt.plot(X[idx == 0, 0], X[idx == 0, 1], 'r.', markerSize=12, label='Cluster 1', alpha=0.2)
    plt.plot(X[idx == 1, 0], X[idx == 1, 1], 'b.', markerSize=12, label='Cluster 2', alpha=0.2)

    x1 = iter_ctrs[:, 0, 0].reshape(-1)
    y1 = iter_ctrs[:, 0, 1].reshape(-1)
    x2 = iter_ctrs[:, 1, 0].reshape(-1)
    y2 = iter_ctrs[:, 1, 1].reshape(-1)
    plt.plot(x1, y1, '-rs', markerSize=5, LineWidth=2, label='Location')
    plt.plot(x2, y2, '-bo', markerSize=5, LineWidth=2, label='NW')
    plt.legend()


def show_digit(X):
    """
    Show a clustering centers image
    :param X: cluster center matrix, returned by kmeans.
    :return:
    """
    w = 20
    h = 20
    num_per_line = 10
    show_line = X.shape[0] // 10

    y = np.zeros((h * show_line, w * num_per_line))
    for i in range(show_line):
        for j in range(num_per_line):
            y[i * h:(i + 1) * h, j * w:(j + 1) * w] = X[i * num_per_line + j, :] \
                .reshape(h, w)
    plt.figure(figsize=(10, 20))
    plt.imshow(y, cmap='gray')
