import math


def sigmoid(x):
    """
    激活函数sigmoid
    :param x:
    :return:
    """
    try:
        y = 1. / (1. + math.exp(-x))
        return y
    except:
        return 1e-5
