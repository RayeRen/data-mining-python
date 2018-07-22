import numpy as np

from utils import add_bias


def linear_regression(x, y):
    """
    Linear Regression.

    :param x: training sample features, P-by-N matrix.
    :param y: training sample labels, 1-by-N row vector.
    :return:
    w: learned perceptron parameters, (P+1)-by-1 column vector.
    """
    x = add_bias(x)
    return np.linalg.inv(x @ x.T) @ x @ y.T
