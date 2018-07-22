import numpy as np

from utils import add_bias


def logistic(x, y):
    """
    LR Logistic Regression.

    :param x: training sample features, P-by-N matrix.
    :param y: training sample labels, 1-by-N row vector.
    :return:
    w: learned parameters, (P+1)-by-1 column vector.
    """
    x = add_bias(x)
    lr = 0.1
    w = np.random.rand(x.shape[0], 1)
    for it in range(100):
        delta_w = np.zeros_like(w)
        for i in range(x.shape[1]):
            delta_w[:, 0] += y[0, i] * x[:, i] / (1 + np.exp(y[0, i] * (w.T @ x[:, i])))
        w += lr * delta_w
    return w
