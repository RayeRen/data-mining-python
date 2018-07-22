import matplotlib.pyplot as plt
import numpy as np


def show_face(X, title=None):
    """
    Show a clustering centers image
    :param X: cluster center matrix, returned by kmeans.
    :return:
    """
    w = 32
    h = 32
    num_per_line = 20
    show_line = 2

    y = np.zeros((h * show_line, w * num_per_line))
    for i in range(show_line):
        for j in range(num_per_line):
            y[i * h:(i + 1) * h, j * w:(j + 1) * w] = X[i * num_per_line + j, :] \
                .reshape(h, w).T
    plt.figure(figsize=(10, 20))
    if title is not None:
        plt.title(title)
    plt.imshow(y, cmap='gray')
