import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt


def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def mkdata(n, noisy=None):
    """
    Generate data set.
    :param n: number of samples. 
    :param noisy: if or not add noise to y.
    :return: 
    x: sample features, P-by-N matrix.
    y: sample labels, 1-by-N row vector.
    w: target function parameters, (P+1)-by-1 column vector.
    """
    r = [-1, 1]
    dim = 2

    x = np.random.rand(dim, n) * (r[1] - r[0]) + r[0]
    while True:
        x_sample = add_bias(np.random.rand(dim, dim) * (r[1] - r[0]) + r[0])
        w = nullspace(x_sample.T)
        y = np.sign(w.T @ add_bias(x))
        if np.all(y):
            break

    if noisy == 'noisy':
        idx = np.random.randint(0, n, n // 10)
        y[0, idx] = -y[0, idx]

    return x, y, w


def plotdata(x, y, w_f, w_g, desc):
    plt.figure()
    if x.shape[0] != 2:
        print("WTF")
        return
    plt.plot(x[0, y[0] == 1], x[1, y[0] == 1], color='r', marker='o', linestyle='')
    plt.plot(x[0, y[0] == -1], x[1, y[0] == -1], color='g', marker='o', linestyle='')
    r = np.linspace(np.min(x[0]), np.max(x[0]), 1000)
    plt.ylim(np.min(x[1]), np.max(x[1]))
    plt.plot(r, (-w_f[1] / w_f[2]) * r - w_f[0] / w_f[2], 'b', linestyle='-', linewidth=2)
    plt.plot(r, (-w_g[1] / w_g[2]) * r - w_g[0] / w_g[2], 'b', linestyle='--', linewidth=2)
    plt.title(desc)


def add_bias(x):
    return np.concatenate([np.ones((1, x.shape[1])), x])
