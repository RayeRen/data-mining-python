from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from knn import knn


def mkdata():
    # Class sizes
    N1 = 200
    N2 = 200
    X = np.concatenate([np.random.randn(2, N1), np.random.randn(2, N2) + 2], axis=1)
    y = np.concatenate([np.zeros((1, N1)), np.ones((1, N2))], axis=1)
    return X, y


def download_checkcodes(prefix="", img_num=100):
    for i in range(0, img_num):
        file_name = "checkcodes/" + prefix + str(i + 1) + ".jpg"
        urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", file_name)


def extract_image(image_file_name):
    """
    EXTRACT_IMAGE Extract features from image
    :param image_file_name: filename of image
    :return: X: 140x5 matrix, 5 digits in an image, each digit is a 140x1 vector.
    """
    m = Image.open(image_file_name)
    m.load()
    m = np.asarray(m, dtype="int32")
    d1 = m[4:18, 4: 14].reshape(140, 1)
    d2 = m[4:18, 13: 23].reshape(140, 1)
    d3 = m[4:18, 22: 32].reshape(140, 1)
    d4 = m[4:18, 31: 41].reshape(140, 1)
    d5 = m[4:18, 40: 50].reshape(140, 1)
    X = [d1, d2, d3, d4, d5]
    return np.concatenate(X, axis=1)


def show_image(X):
    """
    SHOW_IMAGE Show a CAPTCHA image
    :param X: 140x5 matrix, 5 digits in an image, each digit is a 140x1 vector.
    :return:
    """
    num = X.shape[1]
    X = X.reshape(14, 10, num).transpose(0, 2, 1).reshape(14, -1)
    plt.imshow(X, cmap='gray')


def knn_plot(X, y, K):
    ma = ['ko', 'ks']
    fc = np.array([[0, 0, 0], [1, 1, 1]])
    ty = np.unique(y.reshape(-1))

    for i in range(len(ty)):
        pos = (y == ty[i])[0]
        plt.plot(X[0, pos], X[1, pos], ma[i], markerfacecolor=fc[i])

    Xv, Yv = np.meshgrid(
        np.arange(np.min(X[0]), np.max(X[0]), 0.05),
        np.arange(np.min(X[1]), np.max(X[1]), 0.05),
    )
    XX = np.concatenate([Xv.reshape(1, -1), Yv.reshape(1, -1)])
    classes = knn(XX, X, y, K)
    plt.contour(Xv, Yv, classes.reshape(Xv.shape), colors='b')
    plt.title('K = %g' % K)
