import numpy as np


def relu_backprop(in_sensitivity, input):
    """
    The backpropagation process of relu
    :param in_sensitivity: the sensitivity from the upper layer, shape: [number of images, number of outputs in feedforward]
    :param input: the input in feedforward process, shape: same as in_sensitivity
    :return:  the sensitivity to the lower layer, shape: same as in_sensitivity
    """
    img_num, output_num = input.shape
    output_sens = np.zeros((img_num, output_num))

    filter = input > 0
    output_sens[filter] = in_sensitivity[filter]
    output_sens[~filter] = 0
    return output_sens
