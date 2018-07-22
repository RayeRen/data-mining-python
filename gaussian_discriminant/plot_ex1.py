import matplotlib
import numpy as np
from gaussian_pos_prob import gaussian_pos_prob
import seaborn as sns
import matplotlib.pyplot as plt


def plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, pos):
    N = 1000

    X0 = np.random.multivariate_normal(mu0, sigma0, round((1 - phi) * N)).T
    X1 = np.random.multivariate_normal(mu1, sigma1, round(phi * N)).T

    x0 = X0[0]
    y0 = X0[1]
    x1 = X1[0]
    y1 = X1[1]

    if len(x0) == 0:
        xmin = np.min(x1)
        ymin = np.min(y1)
        xmax = np.max(x1)
        ymax = np.max(y1)
    elif len(x1) == 0:
        xmin = np.min(x0)
        ymin = np.min(y0)
        xmax = np.max(x0)
        ymax = np.max(y0)
    else:
        xmin = min(np.min(x0), np.min(x1))
        ymin = min(np.min(y0), np.min(y1))
        xmax = max(np.max(x0), np.max(x1))
        ymax = max(np.max(y0), np.max(y1))

    step = 0.1
    xs, ys = np.meshgrid(
        np.arange(xmin, xmax, step),
        np.arange(ymin, ymax, step)
    )
    xy = np.array([xs.flatten(), ys.flatten()]).T

    sigma = np.array([sigma0, sigma1]).transpose([2, 1, 0])
    mu = np.array([mu0, mu1]).T
    pos_prob = gaussian_pos_prob(xy.T, mu, sigma, np.array([[1 - phi, phi]]))[:, 1]

    image_size = xs.shape
    m_pos_prob = (pos_prob > 0.5).astype('int').reshape(image_size)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "violet", "blue"])

    x_delta = (xmax - xmin) * 0.01
    y_delta = (ymax - ymin) * 0.01

    plt.figure(figsize=(6, 6))
    plt.imshow(m_pos_prob, aspect='auto', interpolation='none',
               extent=[xmin - x_delta, xmax + x_delta, ymin - y_delta, ymax + y_delta], origin='lower', alpha=0.2,
               cmap=cmap)
    plt.title(fig_title)
    plt.plot(x0, y0, marker='o', ls='', alpha=0.8, c='r')
    plt.plot(x1, y1, marker='o', ls='', alpha=0.8, c='b')

    diff = np.abs(pos_prob - 0.5)
    threshold = min(np.sort(diff.flatten())[int(len(diff) / 50)], 0.1)
    bb = xy[(diff <= threshold) & (diff>0)]
    plt.plot(bb[:, 0], bb[:, 1], c='black', marker='o', ls='', alpha=0.8, linewidth=5)
    plt.show()
