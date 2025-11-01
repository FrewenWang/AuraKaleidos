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
if [ "$(uname)" == "Darwin" ]; then
    HOST_OS="mac"
    HOST_ARCH="x86_64"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    HOST_OS="linux"
    HOST_ARCH="x86_64"
elif [ "$(expr substr $(uname -s) 1 10)"=="MINGW32_NT" ]; then
    HOST_OS="windows"
    HOST_ARCH="x86_64"
fi

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
    if [ "$HOST_OS" == "mac" ]; then
        BUILD_TARGET_OS="mac"
        BUILD_TARGET_ARCH="x86_64"
    elif [ "$HOST_OS" == "linux" ]; then
        BUILD_TARGET_OS="linux"
        BUILD_TARGET_ARCH="x86_64"
    elif [ "$HOST_OS" == "windows" ]; then
        BUILD_TARGET_OS="windows"
        BUILD_TARGET_ARCH="x86_64"
    fi
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
  if [ "$NDK_HOME" = "" ]; then
      echo "[===Compiler===] NDK_HOME is not set in environment!!!"
      exit 0
  fi
  TARGET_TOOLCHAIN=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake
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
                 -D ANDROID_PLATFORM=$ANDROID_PLATFORM \
                 -D ANDROID_ARM_NEON=ON \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION=$BUILD_PRODUCTION \
                 -D SOC_VENDOR=$BUILD_SOC_VENDOR \
                 -D CMAKE_TOOLCHAIN_FILE=$TARGET_TOOLCHAIN \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 $BUILD_CMAKE_ARGS \
                 ../..
    elif [ "$TARGET" = "android-arm64-v8a" ]; then
         cmake   -D TARGET_OS=android \
                 -D TARGET_ARCH=arm64-v8a \
                 -D ANDROID_ABI=arm64-v8a \
                 -D ANDROID_PLATFORM=$ANDROID_PLATFORM \
                 -D ANDROID_ARM_NEON=ON \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION=$BUILD_PRODUCTION \
                 -D SOC_VENDOR=$BUILD_SOC_VENDOR \
                 -D CMAKE_TOOLCHAIN_FILE=$TARGET_TOOLCHAIN \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 $BUILD_CMAKE_ARGS \
                 ../..
    elif [ "$TARGET" = "windows-x86_64" ]; then
         cmake   -D TARGET_OS=osx \
                 -D TARGET_ARCH=x86_64 \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION=$BUILD_PRODUCTION \
                 -D SOC_VENDOR=$BUILD_SOC_VENDOR \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 $BUILD_CMAKE_ARGS \
                 ../..
    elif [ "$TARGET" = "linux-x86_64" ]; then
         cmake   -D TARGET_OS=linux \
                 -D TARGET_ARCH=x86_64 \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION=$BUILD_PRODUCTION \
                 -D SOC_VENDOR=$BUILD_SOC_VENDOR \
                 -D CMAKE_INSTALL_PREFIX="$INSTALL_PATH" \
                 $BUILD_CMAKE_ARGS \
                 ../..
    elif [ "$TARGET" = "mac-x86_64" ]; then
         cmake   -D TARGET_OS=mac \
                 -D TARGET_ARCH=x86_64 \
                 -D CMAKE_BUILD_TYPE=$BUILD_TYPE \
                 -D PRODUCTION=$BUILD_PRODUCTION \
                 -D SOC_VENDOR=$BUILD_SOC_VENDOR \
                 -D CMAKE_TOOLCHAIN_FILE=$TARGET_TOOLCHAIN \
                 $BUILD_CMAKE_ARGS \
                 ../..
    fi

    echo "[===Compiler===] make target with 18 threads"
    # make -j 18 VERBOSE=1
     make -j 18
    echo "[===Compiler===] make target success!!!"
    make install
    echo "[===Compiler===] install target success!!!"
    cd -
}

# build target
buildTarget

cd ..


exit 0
