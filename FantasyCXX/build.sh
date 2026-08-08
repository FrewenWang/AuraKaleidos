#!/usr/bin/env bash

#################################################################################
# basic compiler configuration
TARGET_INDEX="0"
BUILD_TYPE="release"
TARGET_TOOLCHAIN=""
HOST_OS="windows"
HOST_ARCH="x86_64"

# custom compiler configuration
BUILD_NAME=""
BUILD_VERSION=""
INSTALL_PATH="./"
BUILD_TARGET_OS=""
BUILD_TARGET_ARCH=""
BUILD_PRODUCTION="xiaomi"
BUILD_SOC_VENDOR="qcom"
BUILD_CMAKE_ARGS=""
EXTRA_MODULES_PATH=""

#################################################################################
# set environment variables

#  set current HOST_OS and HOST_ARCH
case "$(uname -s)" in
    Darwin) HOST_OS="mac" ;;
    Linux) HOST_OS="linux" ;;
    MINGW*|MSYS*|CYGWIN*) HOST_OS="windows" ;;
    *) echo "Unsupported host system: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
    x86_64|amd64) HOST_ARCH="x86_64" ;;
    arm64|aarch64) HOST_ARCH="arm64" ;;
    i386|i686) HOST_ARCH="x86" ;;
    *) HOST_ARCH="$(uname -m)" ;;
esac

#################################################################################
# 0- target os  same as host os
# 1- android-armv7a
# 2- android-armv8a

show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -r, --release           Set build type to Release [default]"
    echo "   -d, --debug             Set build type to Debug"
    echo "   --RelWithDebInfo        Set build type to RelWithDebInfo"
    echo "   -t, --target            Set build target:"
    echo "                              0 - osx or ubuntu,same as host_os and host_arch"
    echo "                              1 - android-armv7a"
    echo "                              2 - android-armv8a"
    echo "   -s, --soc_vendor        Target SOC Vendor"
    echo "   -c, --toolchain         Target compile toolchain"
    echo "   -i, --install           Target install path"
    echo "   -p, --production        Target Production"
    echo "   -h, --help              show help message"
    echo
}

# parse arguments
# 检查某些选项必须有参数
require_arg() {
    if [ -z "$2" ] || [[ "$2" == -* ]]; then
        echo "Error: Option $1 requires an argument."
        exit 1
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        -a)                 require_arg "$1" "$2";    BUILD_CMAKE_ARGS=$2;    shift ;;
        -t|--target)        require_arg "$1" "$2";    TARGET_INDEX=$2;        shift ;;
        -r|--release)       BUILD_TYPE="release" ;;
        -d|--debug)         BUILD_TYPE="debug" ;;
        -i|--install)       require_arg "$1" "$2";    INSTALL_PATH=$2;        shift ;;
        -s|--soc_vendor)    require_arg "$1" "$2";    BUILD_SOC_VENDOR=$2;    shift ;;
        -p|--production)    require_arg "$1" "$2";    BUILD_PRODUCTION=$2;    shift ;;
        -n)                 require_arg "$1" "$2";    BUILD_NAME=$2;          shift ;;
        -v)                 require_arg "$1" "$2";    BUILD_VERSION=$2;       shift ;;
        -e|-extral)         require_arg "$1" "$2";    EXTRA_MODULES_PATH=$2;  shift ;;
        -h|--help)          show_help; exit 0 ;;
        *) echo "Warning: Unknown option $1";;
    esac
    shift
done

case "$TARGET_INDEX" in
0)
    BUILD_TARGET_OS="$HOST_OS"
    BUILD_TARGET_ARCH="$HOST_ARCH"
    ;;
1)
    BUILD_TARGET_OS="android"
    BUILD_TARGET_ARCH="armeabi-v7a"
    ;;
2)
    BUILD_TARGET_OS="android"
    BUILD_TARGET_ARCH="arm64-v8a"
    ;;
esac

TARGET=$BUILD_TARGET_OS-$BUILD_TARGET_ARCH

if [[ $TARGET == *"android"* ]]; then
  # 编译Android版本，需要看一下DNK的环境变量
  AURA_ANDROID_NDK="${ANDROID_NDK_ROOT:-${NDK_HOME:-}}"
  if [ -z "$AURA_ANDROID_NDK" ]; then
      echo "[===Compiler===] ANDROID_NDK_ROOT or NDK_HOME is not set!!!" >&2
      exit 1
  fi
  TARGET_TOOLCHAIN="$AURA_ANDROID_NDK/build/cmake/android.toolchain.cmake"
  ANDROID_PLATFORM="android-34"
fi

echo "[===Compiler===] build target:$TARGET, build type:$BUILD_TYPE, toolchain:$TARGET_TOOLCHAIN"

# create build dir if not exists
if [ ! -d build ]; then
    mkdir -p build
fi
cd build || exit 1


buildTarget(){
    BUILD_DIR="$TARGET-$BUILD_TYPE"
    echo "[===Compiler===] begin build output in (${BUILD_DIR}) "
    if [ ! -d $BUILD_DIR ]; then
        mkdir -p $BUILD_DIR
    fi
    cd $BUILD_DIR || exit 1

    # compile & install
    echo "[===Compiler===] begin cmake target: ${TARGET}"
    if [ "$TARGET" = "android-armeabi-v7a" ]; then
        # 注意：ANDROID_ABI、ANDROID_PLATFORM 需要在这里指定
        cmake    -D TARGET_OS=android \
                 -D TARGET_ARCH=armeabi-v7a \
                 -D ANDROID_ABI=armeabi-v7a \
                 -D ANDROID_PLATFORM="$ANDROID_PLATFORM" \
                 -D ANDROID_ARM_NEON=ON \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION="$BUILD_PRODUCTION" \
                 -D SOC_VENDOR="$BUILD_SOC_VENDOR" \
                 -D CMAKE_TOOLCHAIN_FILE="$TARGET_TOOLCHAIN" \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 "$BUILD_CMAKE_ARGS" \
                 ../..
    elif [ "$TARGET" = "android-arm64-v8a" ]; then
         cmake   -D TARGET_OS=android \
                 -D TARGET_ARCH=arm64-v8a \
                 -D ANDROID_ABI=arm64-v8a \
                 -D ANDROID_PLATFORM="$ANDROID_PLATFORM" \
                 -D ANDROID_ARM_NEON=ON \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION="$BUILD_PRODUCTION" \
                 -D SOC_VENDOR="$BUILD_SOC_VENDOR" \
                 -D CMAKE_TOOLCHAIN_FILE="$TARGET_TOOLCHAIN" \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 "$BUILD_CMAKE_ARGS" \
                 ../..
    elif [[ "$TARGET" == windows-* ]]; then
         cmake   -D TARGET_OS=windows \
                 -D TARGET_ARCH="$BUILD_TARGET_ARCH" \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION="$BUILD_PRODUCTION" \
                 -D SOC_VENDOR="$BUILD_SOC_VENDOR" \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 "$BUILD_CMAKE_ARGS" \
                 ../..
    elif [[ "$TARGET" == linux-* ]]; then
         cmake   -D TARGET_OS=linux \
                 -D TARGET_ARCH="$BUILD_TARGET_ARCH" \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION="$BUILD_PRODUCTION" \
                 -D SOC_VENDOR="$BUILD_SOC_VENDOR" \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 "$BUILD_CMAKE_ARGS" \
                 ../..
    elif [[ "$TARGET" == mac-* ]]; then
         cmake   -D TARGET_OS=mac \
                 -D TARGET_ARCH="$BUILD_TARGET_ARCH" \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION="$BUILD_PRODUCTION" \
                 -D SOC_VENDOR="$BUILD_SOC_VENDOR" \
                 -D CMAKE_TOOLCHAIN_FILE="$TARGET_TOOLCHAIN" \
                 "$BUILD_CMAKE_ARGS" \
                 ../..
    fi

    case "$BUILD_TYPE" in
        debug|Debug) CMAKE_CONFIG="Debug" ;;
        relwithdebinfo|RelWithDebInfo) CMAKE_CONFIG="RelWithDebInfo" ;;
        *) CMAKE_CONFIG="Release" ;;
    esac
    echo "[===Compiler===] build target with CMake"
    cmake --build . --config "$CMAKE_CONFIG" --parallel
    echo "[===Compiler===] build target success!!!"
    cmake --install . --config "$CMAKE_CONFIG"
    echo "[===Compiler===] install target success!!!"
    cd -
}

# build target
buildTarget

cd ..


exit 0
