import glob
import os
import random

import numpy as np

# ===================== 可修改的配置项 (按需改这里即可) =====================
INPUT_DIR = (
    "/data/wzj/ASD量化数据集/all_data_quant"  # 你的原始NHWC格式文件存放目录
)
OUTPUT_DIR = "./input_data"  # bin文件输出目录，固定为input_data
INPUT_TXT_PATH = (
    "./input_data/input_list.txt"  # bin文件输出目录，固定为input_data
)
H, W, CH = 256, 256, 3  # 固定NHWC维度 (1, h, w, ch)
FILE_SUFFIX = [".bin", ".npy", ".raw"]  # 要读取的源文件后缀，按需修改
DTYPE = np.float32  # 数据类型，固定float32，和原脚本一致

RANDOM_SEED = 42  # 随机种子（可修改或注释掉）
SAMPLE_SIZE = 100  # 要抽取的文件数量

# 设置随机种子（保证每次运行抽取结果一致）
random.seed(RANDOM_SEED)
# ==========================================================================


# 创建输出目录，不存在则自动创建
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 遍历目录下所有匹配的文件并排序，保证索引顺序固定不乱序
file_paths = []
for suffix in FILE_SUFFIX:
    file_paths += glob.glob(os.path.join(INPUT_DIR, f"*{suffix}"))
file_paths = sorted(file_paths)  # 关键排序，保证索引0、1、2...对应固定文件


# 2. 随机抽取 SAMPLE_SIZE 个文件（若总数不足则取全部）
if len(file_paths) > SAMPLE_SIZE:
    selected_paths = random.sample(file_paths, SAMPLE_SIZE)
else:
    selected_paths = file_paths
    print(
        f"警告：目录下只有 {len(file_paths)} 个文件，少于 {SAMPLE_SIZE}，将全部处理。"
    )


# 3. 对抽取的文件按索引顺序处理（可再次排序，保持索引与文件名对应）
print(f"共筛选出 {len(file_paths)} 个文件")
selected_paths = sorted(
    selected_paths
)  # 可选：保证生成的 input_0.bin, input_1.bin ... 按文件名排序

# 4. 初始化存储内容
inputs_txt_lines = []  # 存inputs.txt的每行内容

# 5. 遍历文件，按索引读取+转存+写入txt
for input_index, file_path in enumerate(selected_paths):
    if ".npy" in file_path:
        raw_data = np.load(file_path)
        # 强制重塑为标准NHWC格式 (1, H, W, CH)，兼容展平/多维数据
        nhwc_data = raw_data.reshape(1, H, W, CH)
    else:
        # 读取原始文件数据
        raw_data = np.fromfile(file_path, dtype=DTYPE)
        # 强制重塑为标准NHWC格式 (1, H, W, CH)，兼容展平/多维数据
        nhwc_data = raw_data.reshape(1, H, W, CH)

    # 生成带索引的bin文件名 + 保存路径
    bin_filename = f"input_{input_index}.bin"
    bin_save_path = os.path.join(OUTPUT_DIR, bin_filename)
    # 保存为二进制文件，和原脚本tofile用法一致
    nhwc_data.tofile(bin_save_path)

    # ✅ 核心：生成inputs.txt的指定格式内容 input:=input_data/xxx.bin
    inputs_txt_lines.append(f"input:=input_data/{bin_filename}")

    # 打印进度日志
    print(
        f"[{input_index}] 读取: {os.path.basename(file_path)} → 保存: {bin_filename} | 形状: {nhwc_data.shape}"
    )

# 6. 生成核心的 inputs.txt 文件 (当前目录下生成，不是input_data里)
with open(INPUT_TXT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(inputs_txt_lines))

# 打印完成日志
print("\n===== 处理完成！=====")
print(f"✅ 共处理 {len(file_paths)} 个文件")
print(f"✅ bin文件位置: {OUTPUT_DIR}/")
print(f"✅ inputs.txt位置: {os.path.abspath('./inputs.txt')}")
print(
    "✅ inputs.txt格式: input:=input_data/input_索引.bin (完美匹配QNN量化要求)"
)
