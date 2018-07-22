from math import *
from PIL import Image
from pca import pca
import numpy as np


def hack_pca(filename):
    """

    :param filename: input image file name/path
    :return: image without rotation
    """
    img = Image.open(filename)
    a_img = np.asarray(img)
    m = np.array(np.where(a_img != 255))
    vec, first_val = pca(m)
    first_vec = vec[:, 0]
    if first_vec[0] < 0:
        first_vec = -first_vec
    base_vec = np.array([0, 1]).reshape(-1, 1)
    rot_arc = np.arccos(first_vec.T @ base_vec)
    rot_deg = np.sign(rot_arc) * degrees(rot_arc)
    if rot_deg > 90:
        rot_deg = -(180 - rot_deg)
    img.rotate(rot_deg)
