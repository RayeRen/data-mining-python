import scipy.io as sio
import numpy as np

from knn import knn
from utils import extract_image, show_image


def hack(img_name):
    """
    HACK Recognize a CAPTCHA image
    :param img_name: filename of image
    :return: digits: 1x5 matrix, 5 digits in the input CAPTCHA image.
    """
    data = sio.loadmat('hack_data.mat')
    X = data['X']
    y = data['y']
    digits = extract_image(img_name)
    show_image(digits)
    digits = knn(digits, X, y, 20).reshape(-1)
    print('The digits in this CAPTCHA is %d %d %d %d %d' % (digits[0], digits[1], digits[2], digits[3], digits[4]))
