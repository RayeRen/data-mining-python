def get_new_weight_inc(weight_inc, weight, momW, wc, lr, weight_grad):
    """
    Get new increment weight, the update weight policy.
    :param weight_inc: old increment weights
    :param weight: old weights
    :param momW: weight momentum
    :param wc: weight decay
    :param lr: learning rate
    :param weight_grad: weight gradient
    :return: new increment weights
    """
    return momW * weight_inc - wc * lr * weight - lr * weight_grad
