import numpy as np
import matplotlib.pyplot as plt


def show_digit(fea):
    """
    show digits in X
    :param fea:
    :return:
    """
    idx = np.random.permutation(fea.shape[1])
    fea = fea[:, idx[:100]]
    fea = fea.T

    face_w = 28
    face_h = 28
    num_per_line = 20
    show_line = 4

    y = np.zeros((face_h * show_line, face_w * num_per_line))
    for i in range(show_line):
        for j in range(num_per_line):
            y[i * face_h:(i + 1) * face_h, j * face_w:(j + 1) * face_w] = fea[i * num_per_line + j, :] \
                .reshape(face_h, face_w)
    plt.figure(figsize=(10, 20))
    plt.imshow(y, cmap='gray')


def add_bias(x):
    return np.concatenate([np.ones((1, x.shape[1])), x])


def norm(x, mean=None, std=None):
    if mean is None:
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
    std[(std < 0.0001)] = 1
    mean[(std < 0.0001)] = 0
    return (x - mean) / std, mean, std
