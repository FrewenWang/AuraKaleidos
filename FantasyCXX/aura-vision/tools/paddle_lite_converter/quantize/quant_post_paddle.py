import os
import sys
import logging
import paddle
import argparse
import functools
import math
import time
import numpy as np
import paddle.fluid as fluid
import val_reader

sys.path[0] = os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir)
from paddleslim.common import get_logger
from paddleslim.quant import quant_post


batch_size = 10 #32
batch_num = 10 #16
use_gpu = False
model_filename = None
params_filename = None
algo = "KL"
quantizable_op_type = ["conv2d", "depthwise_conv2d", "mul"]


def quantize(model_path, save_path, img_dir):
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    with open(img_dir + "/input_shape.txt") as shape_f:
        shape_str = shape_f.read()
        input_shape = [int(n) for n in shape_str.split()]
        quant_post(executor=exe,
                   model_dir=model_path,
                   model_filename=model_filename,
                   params_filename=params_filename,
                   quantize_model_path=save_path,
                   sample_generator=val_reader.calib_reader_creator(input_shape, img_dir + "/val_list.txt", img_dir, batch_size),
                   batch_size=batch_size,
                   batch_nums=batch_num,
                   algo=algo,
                   quantizable_op_type=quantizable_op_type)


def main(args):
    model_path = args.model_path
    save_path = args.save_path
    img_dir = args.img_dir
    print("model_path   = " + model_path)
    print("save_path    = " + save_path)
    print("img_dir      = " + img_dir)

    quantize(model_path, save_path, img_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", help="image dir")
    parser.add_argument("--model_path", help="the model file path")
    parser.add_argument("--save_path", help="the saved model path")
    args = parser.parse_args()

    main(args)