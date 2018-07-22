import numpy as np


def get_x_distribution(x1, x2, r):
    distribution = np.zeros((2, r[1] - r[0] + 1))
    distribution[0, np.min(x1) - r[0]:np.max(x1) - r[0] + 1] = np.histogram(x1, np.max(x1) - np.min(x1) + 1)[0]
    distribution[1, np.min(x2) - r[0]:np.max(x2) - r[0] + 1] = np.histogram(x2, np.max(x2) - np.min(x2) + 1)[0]
    return distribution
