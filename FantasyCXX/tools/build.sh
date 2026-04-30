#!/usr/bin/env bash

# ============================================================================
# 通用 C++ 跨平台构建系统
# 可复用于不同的 C++ 项目，支持 Android、iOS、macOS、Linux、Windows
# ============================================================================

set -e

# 默认配置
PROJECT_NAME="MyProject"
CMAKE_MIN_VERSION="3.10"
DEFAULT_BUILD_TYPE="release"
DEFAULT_TARGET="host"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
echo_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
echo_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# 显示帮助
show_help() {
    cat << EOF
🎯 通用 C++ 跨平台构建系统

用法: $0 [选项] <目标平台>

目标平台:
    host       - 构建当前平台 (默认)
    android    - Android ARM64
    ios        - iOS ARM64
    macos      - macOS (x86_64 和 ARM64)
    linux      - Linux x86_64
    windows    - Windows x86_64
    all        - 构建所有支持的平台
    clean      - 清理构建文件
    list       - 列出支持的平台

构建选项:
    -t, --type TYPE      构建类型: release|debug (默认: $DEFAULT_BUILD_TYPE)
    -n, --name NAME      项目名称 (默认: $PROJECT_NAME)
    -b, --build-dir DIR  构建目录 (默认: build/)
    -i, --install DIR    安装目录 (默认: install/)
    -j, --jobs NUM       并行构建线程数 (默认: CPU核心数)
    --cmake-args ARGS   额外的 CMake 参数
    --skip-tests         跳过测试构建
    --skip-examples      跳过示例构建
    -v, --verbose        详细输出
    -h, --help           显示此帮助

环境变量:
    ANDROID_NDK_HOME    Android NDK 路径 (Android 构建必需)
    IOS_SDK_PATH        iOS SDK 路径 (iOS 构建时需要)
    BUILD_THREADS       并行构建线程数

示例:
    # 构建当前平台
    $0 host

    # 构建 Android 发布版本
    $0 android --type release

    # 构建所有平台
    $0 all

    # 清理构建
    $0 clean

    # 自定义项目
    $0 android --name MyLib --type debug

EOF
}

# 加载项目配置
load_project_config() {
    local config_file="$PROJECT_ROOT/build.config"

    if [ -f "$config_file" ]; then
        echo_info "加载项目配置: $config_file"
        source "$config_file"
    fi
}

# 检查环境
check_environment() {
    local target="$1"

    # 检查 CMake
    if ! command -v cmake >/dev/null 2>&1; then
        echo_error "CMake 未安装"
        echo_info "请安装 CMake: brew install cmake (macOS) 或 apt install cmake (Linux)"
        exit 1
    fi

    local cmake_version=$(cmake --version | head -1 | cut -d'