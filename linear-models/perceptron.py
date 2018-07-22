import numpy as np

from utils import add_bias


def perceptron(x, y):
    """
    Perceptron Learning Algorithm.

    :param x: training sample features, P-by-N matrix.
    :param y: training sample labels, 1-by-N row vector.
    :return:
    x: learned perceptron parameters, (P+1)-by-1 column vector.
    iter: number of iterations
    """
    it = 0
    x = add_bias(x)
    p, n = x.shape
    w = np.random.rand(p, 1)
    pos = np.where(y * (w.T @ x) < 0)
    while len(pos[0]) > 0:
        w += x[:, pos[1]] @ y[:, pos[1]].T
        pos = np.where(y * (w.T @ x) < 0)
        it += 1

    return w, it
