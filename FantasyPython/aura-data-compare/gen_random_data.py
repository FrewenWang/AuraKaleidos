import numpy as np
import os
import math
from enum import Enum


class PatternType(Enum):
    GRADIENT = 1
    STRIPES = 2
    RANDOM = 3
    CIRCLE = 4
    SINEWAVE = 5
    CHECKERBOARD = 6
    GAUSSIAN = 7
    STEP_FUNCTION = 8


class DataType(Enum):
    U8 = 1
    S8 = 2
    U16 = 3
    S16 = 4
    U32 = 5
    S32 = 6
    F32 = 7
    F64 = 8


class SimpleImageGenerator:
    def __init__(self, width, height, dtype=DataType.U8, pattern=PatternType.GRADIENT):
        self.width = width
        self.height = height
        self.dtype = dtype
        self.pattern = pattern
        self.dtype_map = {
            DataType.U8: np.uint8,
            DataType.S8: np.int8,
            DataType.U16: np.uint16,
            DataType.S16: np.int16,
            DataType.U32: np.uint32,
            DataType.S32: np.int32,
            DataType.F32: np.float32,
            DataType.F64: np.float64
        }

    def generate_image(self):
        dtype_cls = self.dtype_map[self.dtype]
        img_array = np.zeros((self.height, self.width), dtype=dtype_cls)

        # 获取数据范围
        if dtype_cls in (np.float32, np.float64):
            min_val, max_val = -1.0, 1.0
        else:
            info = np.iinfo(dtype_cls)
            min_val, max_val = info.min, info.max

        range_val = max_val - min_val

        # 根据模式生成图像（与之前相同）
        if self.pattern == PatternType.GRADIENT:
            # 水平渐变
            for i in range(self.height):
                progress = i / (self.height - 1)
                value = min_val + range_val * progress
                img_array[i, :] = value

        elif self.pattern == PatternType.STRIPES:
            # 垂直条纹
            stripe_width = max(1, self.width // 10)
            for j in range(self.width):
                stripe = (j // stripe_width) % 2
                img_array[:, j] = max_val if stripe else min_val

        # 其他模式实现...

        return img_array

    def save_raw(self, filename, img_array):
        """保存为原始数据文件（无字节序处理）"""
        img_array.tofile(filename)
        file_size = os.path.getsize(filename)
        print(
            f"保存成功: {filename} | 尺寸: {self.width}x{self.height} | 文件大小: {file_size/(1024**2):.2f} MB")

    def generate_and_save(self, filename):
        img_data = self.generate_image()
        self.save_raw(filename, img_data)


def simple_generate_custom_file():
    """简化版生成自定义图像数据文件"""
    print("简化的图像原始数据文件生成器")
    print("===========================")

    # 获取用户输入
    width = int(input("输入宽度 (默认256): ") or 256)
    height = int(input("输入高度 (默认256): ") or 256)

    print("\n可选数据类型:")
    for i, dtype in enumerate(DataType, 1):
        print(f"{i}. {dtype.name}")
    dtype_idx = int(input("选择数据类型 (默认1): ") or 1) - 1
    dtype = list(DataType)[dtype_idx]

    print("\n可选图案:")
    for i, pattern in enumerate(PatternType, 1):
        print(f"{i}. {pattern.name}")
    pattern_idx = int(input("选择图案 (默认1): ") or 1) - 1
    pattern = list(PatternType)[pattern_idx]

    filename = input("输出文件名 (如: image_data.bin): ") or \
        f"{width}x{height}_{dtype.name.lower()}_{pattern.name.lower()}.bin"

    # 生成并保存
    generator = SimpleImageGenerator(width, height, dtype, pattern)
    generator.generate_and_save(filename)

    print(f"\n文件 {filename} 已成功生成!")


if __name__ == "__main__":
    simple_generate_custom_file()
