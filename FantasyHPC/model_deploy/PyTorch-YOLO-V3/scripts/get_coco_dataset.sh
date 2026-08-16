#!/bin/bash
# -------------------------------------------------------------------
# COCO 2014 数据集下载与解压脚本
# 原始来源: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh
#
# 优化：
#   - 用文件大小快速判断 zip 是否完整（避免 unzip -t 遍历大文件）
#   - 完整则跳过下载直接解压；不完整则 wget -c 断点续传
#   - 下载完成后自动解压，解压后自动清理 zip 节省空间
#   - 支持从任意目录运行（自动定位项目 data/coco 目录）
# -------------------------------------------------------------------
# 自己存储的数据集：
#   - 百度网盘：我的网盘/03.ProgramSpace/20.AI/04.Resource/模型训练数据集/COCO数据集
#
#

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 获取项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COCO_DIR="$PROJECT_DIR/data/coco"
IMAGES_DIR="$COCO_DIR/images"

# 各文件预期的最小字节数（下载完成的文件应不小于此值）
# val2014.zip  ~6.2GB,  train2014.zip ~13GB
declare -A MIN_SIZES=(
    ["val2014.zip"]=6442450944      # 6GB
    ["train2014.zip"]=13958643712   # 13GB
)

echo "=============================================="
echo "  COCO 2014 数据集下载与解压"
echo "=============================================="
echo "数据目录: $COCO_DIR"
echo ""

mkdir -p "$IMAGES_DIR"

# -------------------------------------------------------------------
# 检测 zip 文件是否完整（用文件大小快速判断）
# -------------------------------------------------------------------
is_complete() {
    local file="$1"
    local filename
    filename=$(basename "$file")

    # 无预期大小的文件，只要存在且非空即算完整
    if [ -z "${MIN_SIZES[$filename]}" ]; then
        [ -f "$file" ] && [ -s "$file" ]
        return $?
    fi

    local min_size="${MIN_SIZES[$filename]}"
    local actual_size
    actual_size=$(stat -c%s "$file" 2>/dev/null || echo 0)

    [ "$actual_size" -ge "$min_size" ]
}

# -------------------------------------------------------------------
# 检查目录是否已有有效图片
# -------------------------------------------------------------------
has_images() {
    local dir="$1"
    if [ -d "$dir" ]; then
        local count
        count=$(find "$dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
        [ "$count" -gt 0 ]
        return $?
    fi
    return 1
}

# -------------------------------------------------------------------
# 下载并解压单个 zip（完整流程）
#   $1: URL
#   $2: 目标目录
#   $3: 期望解压后的子目录名（用于检查是否已解压）
# -------------------------------------------------------------------
download_and_extract_zip() {
    local url="$1"
    local dest_dir="$2"
    local expected_subdir="$3"
    local filename
    filename=$(basename "$url")
    local dest_file="$dest_dir/$filename"

    # 1. 如果已解压，直接跳过
    if [ -n "$expected_subdir" ] && has_images "$dest_dir/$expected_subdir"; then
        local count
        count=$(find "$dest_dir/$expected_subdir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} $expected_subdir 已有 $count 张图片，无需下载"
        return 0
    fi

    # 2. 清理空目录
    if [ -d "$dest_dir/$expected_subdir" ]; then
        echo -e "  ${YELLOW}⚠${NC} $expected_subdir 目录为空，清理后重新解压"
        rmdir "$dest_dir/$expected_subdir" 2>/dev/null || true
    fi

    # 3. 判断 zip 文件状态
    if [ -f "$dest_file" ] && is_complete "$dest_file"; then
        # zip 已完整下载 → 跳过下载，直接解压
        local size
        size=$(du -h "$dest_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $filename 已下载完毕 ($size)，跳过下载"
    elif [ -f "$dest_file" ]; then
        # zip 存在但不完整 → 断点续传
        local size
        size=$(du -h "$dest_file" | cut -f1)
        echo -e "  ${YELLOW}↓${NC} $filename 未下载完整 ($size)，断点续传..."
        wget -c -T 30 --show-progress "$url" -P "$dest_dir" || {
            echo -e "  ${RED}✗${NC} $filename 下载失败，请检查网络后重试"
            return 1
        }
        echo -e "  ${GREEN}✓${NC} $filename 续传完成"
    else
        # zip 不存在 → 全新下载
        echo -e "  ${YELLOW}↓${NC} 下载 $filename ..."
        wget -c -T 30 --show-progress "$url" -P "$dest_dir" || {
            echo -e "  ${RED}✗${NC} $filename 下载失败，请检查网络后重试"
            return 1
        }
        echo -e "  ${GREEN}✓${NC} $filename 下载完成"
    fi

    # 4. 解压
    echo -e "  ${YELLOW}↻${NC} 解压 $filename ..."
    if unzip -q "$dest_file" -d "$dest_dir"; then
        echo -e "  ${GREEN}✓${NC} $filename 解压完成"

        # 5. 解压成功后可选清理 zip（大文件默认清理）
        local size
        size=$(du -h "$dest_file" | cut -f1)
        if [ -n "${MIN_SIZES[$filename]}" ]; then
            echo -e "  ${YELLOW}🗑${NC} 清理 $filename ($size) 释放磁盘空间..."
            rm -f "$dest_file"
        fi
    else
        echo -e "  ${RED}✗${NC} $filename 解压失败，文件可能损坏。"
        echo -e "      请执行: rm $dest_file && 重新运行本脚本"
        return 1
    fi
}

# -------------------------------------------------------------------
# 下载普通文件（非 zip，文件存在则跳过）
# -------------------------------------------------------------------
download_if_missing() {
    local url="$1"
    local dest_dir="$2"
    local filename
    filename=$(basename "$url")
    local dest_file="$dest_dir/$filename"

    if [ -f "$dest_file" ] && [ -s "$dest_file" ]; then
        local size
        size=$(du -h "$dest_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $filename 已存在 ($size)，跳过下载"
        return 0
    fi

    # 如果存在空文件则删除
    [ -f "$dest_file" ] && rm -f "$dest_file"

    echo -e "  ${YELLOW}↓${NC} 下载 $filename ..."
    wget -c -T 30 --show-progress "$url" -P "$dest_dir" || {
        echo -e "  ${RED}✗${NC} $filename 下载失败"
        return 1
    }
    echo -e "  ${GREEN}✓${NC} $filename 下载完成"
}

# -------------------------------------------------------------------
# 解压 tgz
# -------------------------------------------------------------------
extract_tgz_if_needed() {
    local tgz_file="$1"
    local dest_dir="$2"
    local expected_subdir="$3"

    if [ -d "$dest_dir/$expected_subdir" ] && [ "$(find "$dest_dir/$expected_subdir" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        local count
        count=$(find "$dest_dir/$expected_subdir" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} $expected_subdir 已有 $count 个文件，跳过解压"
        return 0
    fi

    if [ ! -f "$tgz_file" ]; then
        echo -e "  ${RED}✗${NC} 找不到 $tgz_file"
        return 1
    fi

    echo -e "  ${YELLOW}↻${NC} 解压 $(basename "$tgz_file") ..."
    if tar xzf "$tgz_file" -C "$dest_dir"; then
        echo -e "  ${GREEN}✓${NC} 解压完成"
    else
        echo -e "  ${RED}✗${NC} 解压失败"
        return 1
    fi
}

# -------------------------------------------------------------------
# 解压普通 zip（标注文件等）
# -------------------------------------------------------------------
extract_zip_if_needed() {
    local zip_file="$1"
    local dest_dir="$2"
    local expected_subdir="$3"

    if [ -d "$dest_dir/$expected_subdir" ] && [ "$(find "$dest_dir/$expected_subdir" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        local count
        count=$(find "$dest_dir/$expected_subdir" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✓${NC} $expected_subdir 已有 $count 个文件，跳过解压"
        return 0
    fi

    if [ ! -f "$zip_file" ]; then
        echo -e "  ${RED}✗${NC} 找不到 $zip_file"
        return 1
    fi

    echo -e "  ${YELLOW}↻${NC} 解压 $(basename "$zip_file") ..."
    if unzip -q -o "$zip_file" -d "$dest_dir"; then
        echo -e "  ${GREEN}✓${NC} 解压完成"
    else
        echo -e "  ${RED}✗${NC} 解压失败"
        return 1
    fi
}

# ===================== 主流程 =====================

# 1. 图片下载与解压
echo "--- 图片 ---"
cd "$IMAGES_DIR" || exit 1

download_and_extract_zip \
    "https://pjreddie.com/media/files/val2014.zip" \
    "$IMAGES_DIR" \
    "val2014"

download_and_extract_zip \
    "https://pjreddie.com/media/files/train2014.zip" \
    "$IMAGES_DIR" \
    "train2014"

# 2. 标注与元数据
echo ""
echo "--- 标注与元数据 ---"
cd "$COCO_DIR" || exit 1

download_if_missing "https://pjreddie.com/media/files/instances_train-val2014.zip" "$COCO_DIR"
extract_zip_if_needed "$COCO_DIR/instances_train-val2014.zip" "$COCO_DIR" "annotations"

download_if_missing "https://pjreddie.com/media/files/coco/5k.part" "$COCO_DIR"
download_if_missing "https://pjreddie.com/media/files/coco/trainvalno5k.part" "$COCO_DIR"

download_if_missing "https://pjreddie.com/media/files/coco/labels.tgz" "$COCO_DIR"
extract_tgz_if_needed "$COCO_DIR/labels.tgz" "$COCO_DIR" "labels"

# 3. 生成图片路径列表
echo ""
echo "--- 图片列表 ---"
cd "$COCO_DIR" || exit 1

if [ -f "$COCO_DIR/5k.part" ]; then
    if [ -f "$COCO_DIR/5k.txt" ] && [ -s "$COCO_DIR/5k.txt" ]; then
        echo -e "  ${GREEN}✓${NC} 5k.txt 已存在，跳过生成"
    else
        echo -e "  ${YELLOW}↻${NC} 生成 5k.txt ..."
        paste <(awk "{print \"$IMAGES_DIR\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
        echo -e "  ${GREEN}✓${NC} 5k.txt 生成完成"
    fi
fi

if [ -f "$COCO_DIR/trainvalno5k.part" ]; then
    if [ -f "$COCO_DIR/trainvalno5k.txt" ] && [ -s "$COCO_DIR/trainvalno5k.txt" ]; then
        echo -e "  ${GREEN}✓${NC} trainvalno5k.txt 已存在，跳过生成"
    else
        echo -e "  ${YELLOW}↻${NC} 生成 trainvalno5k.txt ..."
        paste <(awk "{print \"$IMAGES_DIR\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
        echo -e "  ${GREEN}✓${NC} trainvalno5k.txt 生成完成"
    fi
fi

# ===================== 汇总报告 =====================
echo ""
echo "=============================================="
echo "  COCO 数据集状态"
echo "=============================================="

for split in train2014 val2014; do
    if has_images "$IMAGES_DIR/$split"; then
        count=$(find "$IMAGES_DIR/$split" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
        echo "  $split 图片: $count 张"
    else
        echo -e "  $split 图片: ${RED}未下载${NC}"
    fi
done

[ -d "$COCO_DIR/annotations" ] && echo "  标注文件: $(find "$COCO_DIR/annotations" -maxdepth 1 -type f 2>/dev/null | wc -l) 个"
[ -d "$COCO_DIR/labels" ]      && echo "  标签文件: $(find "$COCO_DIR/labels" -maxdepth 1 -type f 2>/dev/null | wc -l) 个"

echo "=============================================="
echo ""

TRAIN_COUNT=$(find "$IMAGES_DIR/train2014" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
VAL_COUNT=$(find "$IMAGES_DIR/val2014" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -gt 0 ] && [ "$VAL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}COCO 数据集已就绪，可以开始训练。${NC}"
else
    echo -e "${YELLOW}数据未就绪，请重新运行本脚本继续。${NC}"
fi
