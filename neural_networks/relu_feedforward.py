import numpy as np


def relu_feedforward(input):
    """
    The feedward process of relu
    :param input: the input, shape: any shape of matrix
    :return: the output, shape: same as input
    """
    return np.maximum(input, 0)
