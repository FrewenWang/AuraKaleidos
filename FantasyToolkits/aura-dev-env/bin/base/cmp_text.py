#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import cv2
import os
import tempfile
import subprocess
import uuid
import csv
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm



# 选择一个支持中文的字体路径
font_path = '/usr/share/fonts/truetype/arphic/ukai.ttc'  # Linux 示例，或者用本机有的中文字体路径
my_font = fm.FontProperties(fname=font_path)

dtype_map = {
    "uint8": np.uint8,
    "u8": np.uint8,
    "uint16": np.uint16,
    "u16": np.uint16,
    "uint32": np.uint32,
    "u32": np.uint32,
    "float16": np.float16,
    "f16": np.float16,
    "float32": np.float32,
    "f32": np.float32,
    "float64": np.float64,
    "f64": np.float64,
}


def adb_pull(path_on_device):
    """从 Android 设备拉取文件到临时目录，并返回本地路径"""
    local_temp_file = os.path.join(tempfile.gettempdir(), f"adb_tmp_{uuid.uuid4().hex}")

    print(f"[ADB] 正在拉取文件: {path_on_device}")
    try:
        subprocess.check_call(["adb", "pull", path_on_device, local_temp_file])
    except subprocess.CalledProcessError:
        raise RuntimeError(f"ADB 拉取失败: {path_on_device}")

    if not os.path.exists(local_temp_file):
        raise RuntimeError(f"ADB 拉取失败，文件未找到: {local_temp_file}")

    print(f"[ADB] 文件已保存到: {local_temp_file}")
    return local_temp_file


def resolve_path(path):
    """如果路径以 adb:/ 开头，则通过 ADB 拉取"""
    if path.startswith("adb:"):
        android_path = path[len("adb:"):]
        return adb_pull(android_path)
    return path


def load_image(filepath, width=None, height=None, dtype=None):
    real_path = filepath
    filepath = resolve_path(filepath)  # 增加 Android 适配

    ext = os.path.splitext(filepath)[-1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        return img

    # RAW Load
    if width is None or height is None or dtype is None:
        raise ValueError("RAW 文件必须指定 width / height / dtype")

    np_dtype = dtype_map[dtype]
    img = np.fromfile(filepath, dtype=np_dtype)

    if img.size != width * height:
        size_bytes = os.path.getsize(filepath)
        raise RuntimeError(f"数据大小不正确，期望: ({width} * {height} * {dtype}) = {width * height * np_dtype().itemsize}, 实际: {size_bytes} 文件路径:{real_path}")

    img = img.reshape((height, width))
    return img


def build_diff_map(img1, img2, threshold):
    diff = np.abs(img1.astype(np.int64) - img2.astype(np.int64))
    diff_map = np.zeros((*img1.shape, 3), dtype=np.uint8)

    same_mask = diff <= threshold
    diff_mask = diff > threshold

    diff_map[same_mask] = (0, 255, 0)  # Green
    diff_map[diff_mask] = (255, 0, 0)  # Red

    return diff_map, diff


def on_submit(text):
    try:
        x_str, y_str = text.strip().split(',')
        x, y = int(x_str), int(y_str)
        if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
            v1 = img1[y, x]
            v2 = img2[y, x]
            d = diff[y, x]
            info_text.set_text(f"坐标({x},{y}): input1={v1}, input2={v2}, diff={d}")
        else:
            info_text.set_text(f"坐标超出范围: ({x},{y})")
    except Exception:
        info_text.set_text("输入格式错误！请用 x,y 形式输入")
    plt.draw()

def on_key(event, axes):
    if event.key == '1':
        axes[0].set_visible(True)
        axes[0].set_position([0.1, 0.1, 0.8, 0.8])  # 占满大部分画布
        axes[1].set_visible(False)
        axes[2].set_visible(False)
    elif event.key == '2':
        axes[0].set_visible(False)
        axes[1].set_visible(True)
        axes[1].set_position([0.1, 0.1, 0.8, 0.8])
        axes[2].set_visible(False)
    elif event.key == '3':
        axes[0].set_visible(False)
        axes[1].set_visible(False)
        axes[2].set_visible(True)
        axes[2].set_position([0.1, 0.1, 0.8, 0.8])
    elif event.key == 'a':
        axes[0].set_visible(True)
        axes[1].set_visible(True)
        axes[2].set_visible(True)

        axes[0].set_position([0.05, 0.1, 0.27, 0.8])
        axes[1].set_position([0.36, 0.1, 0.27, 0.8])
        axes[2].set_position([0.67, 0.1, 0.27, 0.8])
    else:
        # 其他按键忽略
        return
    plt.draw()


def main():
    parser = argparse.ArgumentParser(description="Compare image/raw data visually (support Android ADB)")
    parser.add_argument("--input1", required=True)
    parser.add_argument("--input2", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--save", action="store_true",
                    help="Save raw data of input1/input2 as txt matrices")

    args = parser.parse_args()

    img1 = load_data(args.input1, args.width, args.height, args.dtype)
    img2 = load_data(args.input2, args.width, args.height, args.dtype)



if __name__ == "__main__":
    main()
