import numpy as np


def gaussian_prob(x, mu, sigma):
    p = np.array(
        np.exp(-1 / 2 * (x - mu).T @ np.mat(sigma).I @ (x - mu)) / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    )
    return p[0][0]
