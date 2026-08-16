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
    "int16": np.int16,
    "s16": np.int16,
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


# def on_move(event, img1, img2):
#     if not event.inaxes:
#         return

#     x, y = int(event.xdata), int(event.ydata)
#     if 0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]:
#         v1 = img1[y, x]
#         v2 = img2[y, x]
#         print(f"Pos: ({x}, {y})  input1={v1}  input2={v2}")

def on_move(event, img1, img2, diff, texts, axes):
    if event.inaxes not in axes:
        # 鼠标不在我们关心的子图上，隐藏所有文本
        for txt in texts.values():
            txt.set_visible(False)
        event.canvas.draw_idle()
        return

    ax = event.inaxes
    x, y = int(event.xdata), int(event.ydata)

    if not (0 <= x < img1.shape[1] and 0 <= y < img1.shape[0]):
        for txt in texts.values():
            txt.set_visible(False)
        event.canvas.draw_idle()
        return

    v1 = img1[y, x]
    v2 = img2[y, x]
    d = diff[y, x]

    # 遍历所有文本控件，先隐藏
    for txt in texts.values():
        txt.set_visible(False)

    # 找到当前子图对应的文本，更新显示
    if ax == axes[0]:
        txt = texts['Input1']
    elif ax == axes[1]:
        txt = texts['Input2']
    else:
        txt = texts['Diff']

    txt.set_text(f"Pos: ({y},{x})\ninput1={v1}\ninput2={v2}\ndiff={d}")
    txt.set_visible(True)

    event.canvas.draw_idle()



def save_raw_to_txt_matrix(img, out_path, dtype):
    h, w = img.shape
    with open(out_path, "w") as f:
        f.write(f"# shape: {h} x {w}\n")
        f.write(f"# dtype: {dtype}\n")
        idx = 0
        for y in range(h):
            f.write(f"# y={y}\n")
            row = []
            for x in range(w):
                # row.append(f"({y},{x})[{idx}]={img[y,x]}")
                row.append(f"[{y},{x}]={img[y,x]}")
                idx += 1
            f.write("  ".join(row) + "\n")

def save_raw_to_csv_matrix(img, out_path):
    """
    将 numpy 矩阵保存为纯数字 CSV 矩阵文件
    每行对应图像一行，每列对应像素值
    """
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in img:
            writer.writerow(row.tolist())


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

    img1 = load_image(args.input1, args.width, args.height, args.dtype)
    img2 = load_image(args.input2, args.width, args.height, args.dtype)

    if img1.shape != img2.shape:
        raise RuntimeError("两个输入的尺寸不一致!")

    # save_raw_to_txt_matrix(img1, "image1.txt",args.dtype)
    # save_raw_to_txt_matrix(img2, "image2.txt",args.dtype)

    diff_map, diff = build_diff_map(img1, img2, args.threshold)


    if args.save:
        save_raw_to_csv_matrix(img1, "image1.csv")
        save_raw_to_csv_matrix(img2, "image2.csv")
        save_raw_to_csv_matrix(diff, "diff.csv")
    else:
        print("未传入 --save 参数，跳过保存")


    total_pixels = diff.size
    exceed_count = np.sum(diff > args.threshold)
    exceed_percentage = round((exceed_count / total_pixels) * 100, 2)

    max_diff = np.max(diff)
    max_idx = np.argmax(diff)
    max_pos = np.unravel_index(max_idx, diff.shape)  # 返回 (y, x)

    val_img1 = img1[max_pos[0], max_pos[1]]
    val_img2 = img2[max_pos[0], max_pos[1]]

    info_str = (
        f"超过阈值({args.threshold})像素: {exceed_count}/{total_pixels} = {exceed_percentage}% "
        f"最大差值: ({max_pos[0]},{max_pos[1]})={max_diff} input1[{val_img1}], input2[{val_img2}]"
        # f"平均差值: {np.mean(diff):.2f}"
    )
    print("\n---------  差异分析结果  ---------")
    print(info_str)
    print("----------------------------------\n")


    # 共创建 1×3=3 个子图
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    axes[0].set_title("Input1")
    axes[0].imshow(img1, cmap='gray')
    axes[0].axis("off")

    axes[1].set_title("Input2")
    axes[1].imshow(img2, cmap='gray')
    axes[1].axis("off")

    axes[2].set_title("Diff (Green=Same, Red=Different)")
    axes[2].imshow(diff_map)
    axes[2].axis("off")

    plt.tight_layout()

    # Cursor Hover
    # cursor0 = Cursor(axes[0], useblit=True, color='yellow', linewidth=1)
    # fig.canvas.mpl_connect("motion_notify_event",
    #                        lambda event: on_move(event, img1, img2))

    # cursor1 = Cursor(axes[1], useblit=True, color='yellow', linewidth=1)
    # fig.canvas.mpl_connect("motion_notify_event",
    #                        lambda event: on_move(event, img1, img2))

    texts = {
    'Input1': axes[0].text(0.05, 0.95, "", transform=axes[0].transAxes,
                          fontsize=10, color='yellow', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5)),
    'Input2': axes[1].text(0.05, 0.95, "", transform=axes[1].transAxes,
                          fontsize=10, color='yellow', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5)),
    'Diff':   axes[2].text(0.05, 0.95, "", transform=axes[2].transAxes,
                          fontsize=10, color='yellow', verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5)),
    }

    # cursor0 = Cursor(axes[0], useblit=True, color='yellow', linewidth=1)
    # cursor1 = Cursor(axes[1], useblit=True, color='yellow', linewidth=1)
    # cursor2 = Cursor(axes[2], useblit=True, color='yellow', linewidth=1)
    fig.canvas.mpl_connect("motion_notify_event",
                           lambda event: on_move(event, img1, img2, diff, texts, axes))

    fig.canvas.mpl_connect('key_press_event', lambda event: on_key(event, axes))

    # 在窗口底部显示差异信息
    info_text = fig.text(0.5, 0.02, info_str, ha='center', va='bottom', fontsize=15,
                        bbox=dict(facecolor='black', alpha=0.7, pad=5), color='white', fontproperties=my_font)

    # 坐标搜索框
    # axbox = plt.axes([0.15, 0.01, 0.3, 0.04])
    # text_box = TextBox(axbox, 'postion (x,y): ', initial="0,0")
    # text_box.on_submit(on_submit)

    # info_text = fig.text(0.6, 0.02, '', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))


    plt.show()


if __name__ == "__main__":
    main()
