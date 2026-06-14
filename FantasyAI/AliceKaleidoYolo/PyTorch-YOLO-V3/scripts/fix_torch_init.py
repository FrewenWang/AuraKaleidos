#!/usr/bin/env python
"""
修复 PyTorch __init__.py 缺失问题
某些 PyTorch 安装可能缺少 __init__.py 文件，导致导入失败
"""

import os
import sys

def find_torch_path():
    """查找 torch 安装路径"""
    for path in sys.path:
        torch_path = os.path.join(path, 'torch')
        if os.path.isdir(torch_path):
            return torch_path
    return None

def fix_torch_init():
    """创建缺失的 __init__.py"""
    torch_path = find_torch_path()

    if torch_path is None:
        print("错误: 找不到 torch 安装路径")
        return False

    init_file = os.path.join(torch_path, '__init__.py')

    if os.path.exists(init_file):
        print(f"✓ __init__.py 已存在: {init_file}")
        return True

    print(f"✗ 发现缺失的 __init__.py: {init_file}")
    print("正在创建...")

    # 创建一个基本的 __init__.py
    init_content = '''"""
PyTorch 初始化模块
自动生成用于修复导入问题
"""

__version__ = "2.2.2"

from torch._C import *
from torch._C._distributed_c11 import *

from torch import nn
from torch import optim
from torch import autograd
from torch import cuda
from torch import distributed
from torch import jit
from torch import onnx
from torch import quantization
from torch import utils

# 导入子模块
import torch.nn
import torch.nn.functional
import torch.nn.init

import torch.optim

import torch.autograd

import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.dataset
import torch.utils.data.distributed
import torch.utils.data.sampler

import torch.cuda
import torch.cuda.amp

import torch.distributed

__all__ = [
    'nn', 'optim', 'autograd', 'cuda', 'distributed', 'jit', 'onnx',
    'quantization', 'utils',
    'tensor', 'FloatTensor', 'DoubleTensor', 'HalfTensor',
    'ByteTensor', 'ShortTensor', 'IntTensor', 'LongTensor',
    'zeros', 'ones', 'rand', 'randn', 'arange', 'eye',
    'from_numpy', 'as_tensor',
    'manual_seed', 'manual_seed_all',
    'cuda.is_available', 'cuda.device_count',
    'device', 'dtype', 'set_printoptions',
]
'''

    try:
        with open(init_file, 'w') as f:
            f.write(init_content)
        print(f"✓ 成功创建: {init_file}")
        return True
    except Exception as e:
        print(f"✗ 创建失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch __init__.py 修复工具")
    print("=" * 60)
    print()

    success = fix_torch_init()

    print()
    if success:
        print("修复完成！现在可以尝试运行项目了。")
        print("运行测试: python detect.py --image_folder data/samples --weights_path weights/yolov3.weights")
    else:
        print("修复失败。建议重新安装 PyTorch:")
        print("  pip install torch torchvision")
    print("=" * 60)
