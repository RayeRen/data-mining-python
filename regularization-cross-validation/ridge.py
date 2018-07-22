import numpy as np

from utils import add_bias


def ridge(x, y, lamb):
    """
    Ridge Regression.
    :param x: training sample features, P-by-N matrix.
    :param y: training sample labels, 1-by-N row vector.
    :param lamb: regularization parameter.
    :return: w: learned parameters, (P+1)-by-1 column vector.
    """
    x = add_bias(x)
    return np.linalg.inv(x @ x.T + lamb * np.eye(x.shape[0])) @ x @ y.T
