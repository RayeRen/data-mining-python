import numpy as np


def softmax_loss(input, label):
    """
    The softmax loss computing process
    :param input: the output of previous layer, shape: [number of images, number of kinds of labels]
    :param label: the ground true of these images, shape: [1, number of images]
    :return: 
    loss: the average loss, scale variable
    accuracy: the accuracy of the classification
    sentivity: the sentivity for input, shape: [number of images, number of kinds of labels]
    """
    label = label - 1

    n, k = input.shape
    input = input - np.max(input, axis=1, keepdims=True)
    h = np.exp(input)
    total = np.sum(h, axis=1, keepdims=True)
    probs = h / total
    idx = [np.arange(n), label[0]]
    loss = -np.sum(np.log(probs[idx])) / n

    max_idx = np.argmax(probs, 1)
    accuracy = np.sum(max_idx == label) / n
    sensitivity = np.zeros((n, k))
    sensitivity[idx] = -1
    sensitivity = sensitivity + probs

    return loss, accuracy, sensitivity
