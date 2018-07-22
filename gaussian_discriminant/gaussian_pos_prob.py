import numpy as np
from tqdm import tqdm

from gaussian_prob import gaussian_prob


def gaussian_pos_prob(x, mu, sigma, phi):
    """
    Posterior probability of GDA. compute the posterior probability of given N data points X using Gaussian
    Discriminant Analysis where the K gaussian distributions are specified by Mu, Sigma and Phi.

    :param x: M-by-N matrix, N data points of dimension M.
    :param mu: M-by-K matrix, mean of K Gaussian distributions.
    :param sigma: M-by-M-by-K matrix (yes, a 3D matrix), variance matrix of K Gaussian distributions.
    :param phi:  1-by-K matrix, prior of K Gaussian distributions.
    :return: N-by-K matrix, posterior probability of N data points with in K Gaussian distributions.
    """
    n = x.shape[1]
    k = phi.shape[1]
    p = np.zeros((n, k))
    for i in tqdm(range(n)):
        g = []
        for j in range(k):
            g.append(gaussian_prob(x[:, i], mu[:, j], sigma[:, :, j]) * phi[0, j])
        g = np.array(g)
        s = np.sum(g)
        for j in range(k):
            p[i][j] = g[j] / s
    return p
