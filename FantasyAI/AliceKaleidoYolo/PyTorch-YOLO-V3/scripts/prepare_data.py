#!/usr/bin/env python
"""
数据准备脚本
解决 COCO 数据集目录结构问题，将标签文件移动到正确位置
"""

import os
import shutil
from pathlib import Path

def prepare_coco_data():
    """准备 COCO 数据集，确保目录结构正确"""

    # 定义路径
    coco_dir = Path("data/coco")
    images_dir = coco_dir / "images"
    labels_source = coco_dir / "labels" / "labels"  # 源标签目录（嵌套）
    labels_target = coco_dir / "labels"  # 目标标签目录

    print("=" * 60)
    print("COCO 数据集准备")
    print("=" * 60)

    # 检查图像目录
    print("\n[1] 检查图像目录...")
    train_dir = images_dir / "train2014"
    val_dir = images_dir / "val2014"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print(f"  ⚠️  训练图像目录为空或不存在: {train_dir}")
        print("  请下载 COCO 2014 训练图像并解压到该目录")
        print("  下载地址: http://cocodataset.org/#download")
    else:
        train_count = len(list(train_dir.glob("*.jpg")))
        print(f"  ✓ 训练图像目录存在，包含 {train_count} 张图像")

    if not val_dir.exists() or not any(val_dir.iterdir()):
        print(f"  ⚠️  验证图像目录为空或不存在: {val_dir}")
        print("  请下载 COCO 2014 验证图像并解压到该目录")
    else:
        val_count = len(list(val_dir.glob("*.jpg")))
        print(f"  ✓ 验证图像目录存在，包含 {val_count} 张图像")

    # 处理标签目录结构
    print("\n[2] 检查标签目录结构...")

    if labels_source.exists():
        print(f"  发现嵌套标签目录: {labels_source}")
        print(f"  正在移动标签文件到: {labels_target}")

        # 确保目标目录存在
        labels_target.mkdir(parents=True, exist_ok=True)

        # 移动 train2014 标签
        train_source = labels_source / "train2014"
        train_target = labels_target / "train2014"

        if train_source.exists():
            if not train_target.exists():
                shutil.move(str(train_source), str(train_target))
                print(f"  ✓ 移动训练标签: {train_source} -> {train_target}")
            else:
                print(f"  ✓ 训练标签目录已存在: {train_target}")

        # 移动 val2014 标签
        val_source = labels_source / "val2014"
        val_target = labels_target / "val2014"

        if val_source.exists():
            if not val_target.exists():
                shutil.move(str(val_source), str(val_target))
                print(f"  ✓ 移动验证标签: {val_source} -> {val_target}")
            else:
                print(f"  ✓ 验证标签目录已存在: {val_target}")

        # 尝试删除空的嵌套目录
        try:
            if labels_source.exists() and not any(labels_source.iterdir()):
                labels_source.rmdir()
                print(f"  ✓ 删除空目录: {labels_source}")
            # 尝试删除父目录
            parent = labels_source.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
                print(f"  ✓ 删除空目录: {parent}")
        except Exception as e:
            print(f"  ℹ️  无法删除空目录（可忽略）: {e}")

    else:
        print("  ✓ 标签目录结构正确")

    # 验证最终结构
    print("\n[3] 验证最终目录结构...")
    print(f"  {coco_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train2014/  ({len(list((images_dir / 'train2014').glob('*.jpg')))} 张图像)")
    print(f"    │   └── val2014/    ({len(list((images_dir / 'val2014').glob('*.jpg')))} 张图像)")
    print(f"    ├── labels/")
    print(f"    │   ├── train2014/  ({len(list((labels_target / 'train2014').glob('*.txt')))} 个标签)")
    print(f"    │   └── val2014/    ({len(list((labels_target / 'val2014').glob('*.txt')))} 个标签)")
    print(f"    ├── trainvalno5k.txt")
    print(f"    └── 5k.txt")

    # 检查数据列表文件
    print("\n[4] 检查数据列表文件...")
    train_list = coco_dir / "trainvalno5k.txt"
    val_list = coco_dir / "5k.txt"

    if train_list.exists():
        with open(train_list) as f:
            lines = f.readlines()
        print(f"  ✓ 训练列表: {len(lines)} 个样本")
    else:
        print(f"  ⚠️  训练列表不存在: {train_list}")

    if val_list.exists():
        with open(val_list) as f:
            lines = f.readlines()
        print(f"  ✓ 验证列表: {len(lines)} 个样本")
    else:
        print(f"  ⚠️  验证列表不存在: {val_list}")

    print("\n" + "=" * 60)
    print("准备完成！")
    print("=" * 60)

if __name__ == "__main__":
    # 百度网盘：通过网盘分享的文件：train2014.zip
    # COCO数据集: 链接: https://pan.baidu.com/s/1smesLlju2Kb4Clxs1dvzBw 提取码: c2iw 
    prepare_coco_data()
