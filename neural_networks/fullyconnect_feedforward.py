import numpy as np


def fullyconnect_feedforward(input, weight, bias):
    """
    The feedward process of fullyconnect
    :param input: the inputs, shape: [number of images, number of inputs]
    :param weight: the weight matrix, shape: [number of inputs, number of outputs]
    :param bias: the bias, shape: [number of outputs, 1]
    :return: the output of this layer, shape: [number of images, number of outputs]
    """
    return input @ weight + bias.reshape(1, -1)
