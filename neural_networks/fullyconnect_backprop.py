import numpy as np


def fullyconnect_backprop(in_sensitivity, input, weight):
    """
    The backpropagation process of fullyconnect
    :param in_sensitivity: the sensitivity from the upper layer, shape: [number of images, number of outputs in feedforward]
    :param input: the input in feedforward process, shape: [number of images, number of inputs in feedforward]
    :param weight: the weight matrix of this layer, shape: [number of inputs in feedforward, number of outputs in feedforward]
    :return:
    weight_grad: the gradient of the weights, shape [number of inputs in feedforward, number of outputs in feedforward]
    out_sensitivity: the sensitivity to the lower layer, shape: [number of images, number of inputs in feedforward]
    """
    input_num, output_num = weight.shape
    img_num = input.shape[0]
    weight_grad = np.zeros((input_num, output_num))
    bias_grad = np.zeros((output_num, 1))
    out_sensitivity = np.zeros((img_num, input_num))
    for i in range(img_num):
        out_sensitivity[i] = in_sensitivity[i] @ weight.T
        weight_grad += input[i].reshape(input_num, 1) @ in_sensitivity[i].reshape(1, output_num)
        bias_grad += in_sensitivity[i].reshape(output_num, 1)
    weight_grad /= img_num
    bias_grad /= img_num
    return weight_grad, bias_grad, out_sensitivity
