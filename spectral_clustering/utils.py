import matplotlib.pyplot as plt


def cluster_plot(X, idx):
    """
    CLUSTER_PLOT show clustering results
    :param X: data point features, n-by-p maxtirx.
    :param idx: data point cluster labels, n-by-1 vector.
    :return:
    """
    plt.plot(X[idx == 0, 0], X[idx == 0, 1], 'r.', label='Cluster 1')
    plt.plot(X[idx == 1, 0], X[idx == 1, 1], 'b.', label='Cluster 1')
    plt.legend()
