#!/bin/bash
# ---------------------------------------------------------------------
# PyTorch-YOLO-V3 环境检查与资源下载脚本
# 自动检测运行环境是否满足要求，并下载所需的预训练权重和 COCO 数据集
# ---------------------------------------------------------------------

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WEIGHTS_DIR="$PROJECT_DIR/weights"

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

echo "=============================================="
echo "  PyTorch-YOLO-V3 环境检查"
echo "=============================================="
echo ""

# -------------------------------------------------------------------
# 1. Python 版本检查
# -------------------------------------------------------------------
check_python() {
    echo -n "[检查] Python 版本 ... "

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}未找到 python3${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 6 ]); then
        echo -e "${RED}版本 $PY_VERSION（需要 >= 3.6）${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi

    echo -e "${GREEN}Python $PY_VERSION ✓${NC}"
    PASS_COUNT=$((PASS_COUNT + 1))
}

# -------------------------------------------------------------------
# 2. Python 依赖包检查
# -------------------------------------------------------------------
check_python_package() {
    local pkg_name="$1"
    local import_name="${2:-$1}"

    echo -n "[检查] Python 包 $pkg_name ... "

    if python3 -c "import $import_name" 2>/dev/null; then
        local ver=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', '已安装'))" 2>/dev/null)
        echo -e "${GREEN}$ver ✓${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
        return 0
    else
        echo -e "${RED}未安装${NC}"
        echo -e "       ${YELLOW}请执行: pip install $pkg_name${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

check_python_packages() {
    echo ""
    echo "--- Python 依赖包 ---"
    check_python_package "torch"
    check_python_package "torchvision"
    check_python_package "numpy"
    check_python_package "opencv-python" "cv2"
    check_python_package "Pillow" "PIL"
    check_python_package "matplotlib"
    check_python_package "tqdm"
}

# -------------------------------------------------------------------
# 3. CUDA 可用性检查（可选，信息提示）
# -------------------------------------------------------------------
check_cuda() {
    echo ""
    echo "--- GPU/CUDA 状态 ---"
    echo -n "[检查] CUDA 可用性 ... "

    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_DEVICE_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        CUDA_DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "未知")
        echo -e "${GREEN}可用（$CUDA_DEVICE_COUNT 个设备）✓${NC}"
        echo "       设备名称: $CUDA_DEVICE_NAME"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "${YELLOW}不可用 — 将使用 CPU 推理${NC}"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi
}

# -------------------------------------------------------------------
# 4. 权重文件检查与下载
# -------------------------------------------------------------------
check_and_download_weights() {
    echo ""
    echo "--- 权重文件 ---"

    local weights=(
        "yolov3.weights"
        "yolov3-tiny.weights"
        "darknet53.conv.74"
    )

    local all_exist=true
    for w in "${weights[@]}"; do
        if [ -f "$WEIGHTS_DIR/$w" ]; then
            local size=$(du -h "$WEIGHTS_DIR/$w" | cut -f1)
            echo -e "  ${GREEN}✓${NC} $w ($size)"
        else
            echo -e "  ${YELLOW}✗${NC} $w（缺失）"
            all_exist=false
        fi
    done

    if [ "$all_exist" = true ]; then
        echo ""
        echo -e "${GREEN}所有权重文件已就绪。${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo ""
        echo -e "${YELLOW}部分权重文件缺失，将自动下载...${NC}"
        echo ""

        if [ -f "$WEIGHTS_DIR/download_weights.sh" ]; then
            echo "执行: bash $WEIGHTS_DIR/download_weights.sh"
            echo "----------------------------------------------"
            (cd "$WEIGHTS_DIR" && bash download_weights.sh)
            echo "----------------------------------------------"

            # 验证下载
            local download_ok=true
            for w in "${weights[@]}"; do
                if [ -f "$WEIGHTS_DIR/$w" ]; then
                    echo -e "  ${GREEN}✓${NC} $w 下载完成"
                else
                    echo -e "  ${RED}✗${NC} $w 下载失败"
                    download_ok=false
                fi
            done

            if [ "$download_ok" = true ]; then
                PASS_COUNT=$((PASS_COUNT + 1))
            else
                FAIL_COUNT=$((FAIL_COUNT + 1))
            fi
        else
            echo -e "${RED}错误: 找不到 download_weights.sh 脚本${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
}

# -------------------------------------------------------------------
# 5. COCO 数据集检查与下载
# -------------------------------------------------------------------
check_and_download_coco() {
    echo ""
    echo "--- COCO 数据集 ---"

    local IMAGES_DIR="$PROJECT_DIR/data/coco/images"
    local COCO_DIR="$PROJECT_DIR/data/coco"

    # 统计已有图片数量
    local train_count=0
    local val_count=0
    if [ -d "$IMAGES_DIR/train2014" ]; then
        train_count=$(find "$IMAGES_DIR/train2014" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
    fi
    if [ -d "$IMAGES_DIR/val2014" ]; then
        val_count=$(find "$IMAGES_DIR/val2014" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | wc -l)
    fi

    echo "  train2014 图片: $train_count 张"
    echo "  val2014 图片:   $val_count 张"

    # 检查标签和标注
    local labels_ok=false
    local annotations_ok=false
    local part_files_ok=false

    if [ -d "$COCO_DIR/labels" ] && [ "$(find "$COCO_DIR/labels" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} labels 标签"
        labels_ok=true
    else
        echo -e "  ${YELLOW}✗${NC} labels 标签（缺失）"
    fi

    if [ -d "$COCO_DIR/annotations" ] && [ "$(find "$COCO_DIR/annotations" -maxdepth 1 -type f 2>/dev/null | wc -l)" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} annotations 标注"
        annotations_ok=true
    else
        echo -e "  ${YELLOW}✗${NC} annotations 标注（缺失）"
    fi

    if [ -f "$COCO_DIR/5k.part" ] && [ -f "$COCO_DIR/trainvalno5k.part" ]; then
        echo -e "  ${GREEN}✓${NC} 图片列表分区文件"
        part_files_ok=true
    else
        echo -e "  ${YELLOW}✗${NC} 图片列表分区文件（缺失）"
    fi

    # 判断数据集是否完整
    local EXPECTED_TRAIN=82783   # COCO train2014 图片数
    local EXPECTED_VAL=40504     # COCO val2014 图片数

    if [ "$train_count" -ge "$EXPECTED_TRAIN" ] && [ "$val_count" -ge "$EXPECTED_VAL" ] && $labels_ok && $annotations_ok && $part_files_ok; then
        echo ""
        echo -e "${GREEN}COCO 数据集已完整就绪。${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [ "$train_count" -gt 0 ] || [ "$val_count" -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}COCO 数据集部分存在（训练集 $train_count/$EXPECTED_TRAIN，验证集 $val_count/$EXPECTED_VAL）${NC}"
        echo -e "${YELLOW}将自动补充下载缺失数据...${NC}"
        WARN_COUNT=$((WARN_COUNT + 1))
        run_coco_download
    else
        echo ""
        echo -e "${YELLOW}COCO 数据集未下载，将自动开始下载...${NC}"
        WARN_COUNT=$((WARN_COUNT + 1))
        run_coco_download
    fi
}

run_coco_download() {
    local download_script="$SCRIPT_DIR/get_coco_dataset.sh"

    if [ -f "$download_script" ]; then
        echo ""
        echo "执行: bash $download_script"
        echo "----------------------------------------------"
        bash "$download_script"
        local dl_result=$?
        echo "----------------------------------------------"

        if [ $dl_result -eq 0 ]; then
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "${RED}COCO 数据集下载未完全成功，可重新运行本脚本继续${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo -e "${RED}错误: 找不到下载脚本 $download_script${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# -------------------------------------------------------------------
# 6. 配置文件检查
# -------------------------------------------------------------------
check_configs() {
    echo ""
    echo "--- 配置文件 ---"

    local configs=(
        "config/yolov3.cfg"
        "config/yolov3-tiny.cfg"
        "config/coco.data"
        "data/coco.names"
    )

    for c in "${configs[@]}"; do
        echo -n "  $c ... "
        if [ -f "$PROJECT_DIR/$c" ]; then
            echo -e "${GREEN}✓${NC}"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "${RED}缺失${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
}

# -------------------------------------------------------------------
# 汇总报告
# -------------------------------------------------------------------
print_summary() {
    echo ""
    echo "=============================================="
    echo "  检查完成"
    echo "=============================================="
    echo -e "  通过: ${GREEN}$PASS_COUNT${NC}"
    echo -e "  警告: ${YELLOW}$WARN_COUNT${NC}"
    echo -e "  失败: ${RED}$FAIL_COUNT${NC}"
    echo "=============================================="

    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo ""
        echo -e "${RED}存在未满足的依赖，请根据上述提示修复后重新运行。${NC}"
        echo ""
        echo "如果缺少 Python 包，可以尝试："
        echo "  pip install torch torchvision numpy opencv-python Pillow matplotlib tqdm"
        exit 1
    else
        echo ""
        echo -e "${GREEN}环境检查全部通过，可以开始使用 PyTorch-YOLO-V3。${NC}"
        echo ""
        echo "-------- 快速开始 --------"
        echo ""
        echo "  # 目标检测（图片需放到 data/samples/）"
        echo "  python detect.py --image_folder data/samples --weights_path weights/yolov3.weights"
        echo ""
        echo "  # 使用轻量模型"
        echo "  python detect.py --image_folder data/samples --model_def config/yolov3-tiny.cfg --weights_path weights/yolov3-tiny.weights"
        echo ""
        echo "  # 训练模型（需要 COCO 数据集）"
        echo "  python train.py --model_def config/yolov3.cfg --data_config config/coco.data --pretrained_weights weights/darknet53.conv.74"
        echo ""
        echo "  详细说明见 scripts/get_coco_dataset.sh"
    fi
}

# -------------------------------------------------------------------
# 主流程
# -------------------------------------------------------------------
main() {
    check_python
    check_python_packages
    check_cuda
    check_configs
    check_and_download_weights
    check_and_download_coco
    print_summary
}

main
