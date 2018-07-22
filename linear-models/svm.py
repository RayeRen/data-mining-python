import numpy as np
from utils import add_bias
from qpsolvers import solve_qp


def svm(x, y):
    """
    SVM Support vector machine.

    :param x: training sample features, P-by-N matrix.
    :param y: training sample labels, 1-by-N row vector.
    :return:
    w:    learned perceptron parameters, (P+1)-by-1 column vector.
    num:  number of support vectors
    """
    x = add_bias(x)
    w = solve_qp(np.eye(3, 3), np.zeros((3, 1)), -np.diag(y[0]) @ x.T, -np.ones((y.shape[1], 1)), solver='cvxopt')
    t = (w.T @ x) * y
    return w, np.sum(np.abs(t - 1) < 0.01)
