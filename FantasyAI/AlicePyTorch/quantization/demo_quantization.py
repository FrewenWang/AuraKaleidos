def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    """ 计算
    :param min_val:  输入的最小值
    :param max_val:  输入的最大值
    :param num_bits: 8比特位量化
    :return:
    """
    qmin = 0.  # 量化之后的最小值
    qmax = 2. ** num_bits - 1.  # 量化之后的最大值2^8-1
    # S=(rmax-rmin)/(qmax-qmin) 求出缩放系数scale
    scale = float((max_val - min_val) / (qmax - qmin))
    # Z=round(qmax-rmax/scale) 求出zero point
    zero_point = qmax - max_val / scale
    # 如果zero_point小于
    if zero_point < qmin:
        zero_point = qmin
    elif zero_point > qmax:
        zero_point = qmax
    #
    zero_point = int(zero_point)
    # 返回求得的scale, zero_point
    return scale, zero_point


def quantize_tensor(x, scale, zero_point, num_bits=8, signed=False):
    """
    进行tensor的量化结果
    :param x:               输入推理值x
    :param scale:           缩放系数
    :param zero_point:      0值
    :param num_bits:        8比特位量化
    :param signed:
    :return:
    """
    if signed:
        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1
    else:
        qmin = 0.
        qmax = 2. ** num_bits - 1.
    # q=round(r/S+Z)
    q_x = zero_point + x / scale

    q_x.clamp_(qmin, qmax).round_()
    # 由于pytorch不支持int类型的运算，因此我们还是用float来表示整数
    return q_x.float()


def dequantize_tensor(q_x, scale, zero_point):
    """
    反量化的的步骤
    :param q_x: 量化之后的技术
    :param scale: 权重的量化系数
    :param zero_point: 0值
    :return: 量化之前的结果
    """
    # 反量化： 浮点数 = scale*(quantize - ZeroPoint)
    # r=S(q-Z)
    return scale * (q_x - zero_point)


class QParam:
    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.scale = None
        self.zero_point = None
        self.min = None
        self.max = None

    def update(self, tensor):
        if self.max is None or self.max < tensor.max():
            self.max = tensor.max()
        self.max = 0 if self.max < 0 else self.max

        if self.min is None or self.min > tensor.min():
            self.min = tensor.min()
        self.min = 0 if self.min > 0 else self.min

    def quantize_tensor(self, tensor):
        return quantize_tensor(tensor, self.scale, self.zero_point, num_bits=self.num_bits)

    def dequantize_tensor(self, q_x):
        return dequantize_tensor(q_x, self.scale, self.zero_point)
