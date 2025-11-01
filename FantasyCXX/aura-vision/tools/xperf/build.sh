#!/usr/bin/env bash

# NDK
# 通过环境变量或手动设置ANDROID_NDK_HOME路径
export ANDROID_NDK_HOME=${ANDROID_NDK_HOME}

# default setting
TARGET_INDEX="0"
BUILD_TYPE="release"
use_external_model=false
use_external_encrypt=false
build_aar=false

ENABLE_SNPE=OFF
ENABLE_QNN=ON

# 0-osx
# 1-android-armv7a
# 2-android-armv8a
# 3-android-x86_64
# 4-android-x86

show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -t, --target            Set build target, 0-osx, 1-android-armv7a, 2-android-armv8a, 3-android-x86_64, 4-android-x86"
    echo "   -s, --snpe              if build with snpe"
    echo "   -q, --qnn               if build with qnn"
    echo "   -h, --help              show help message"
    echo
}

# parse arguments
while [ $# != 0 ]
do
  case "$1" in
    -t)
        TARGET_INDEX=$2
        shift
        ;;
    --target)
        TARGET_INDEX=$2
        shift
        ;;
    -s)
        ENABLE_SNPE=ON
        ;;
    -q)
        ENABLE_QNN=ON
        ;;
    --snpe)
        ENABLE_SNPE=ON
        ;;
    -h)
        show_help
        exit 1
        ;;
    --help)
    show_help
        exit 1
        ;;
    *)
        ;;
  esac
  shift
done

case "$TARGET_INDEX" in
0)
    if [ "$(uname)" == "Darwin" ]; then
      TARGET="osx-x86_64"
      SYSTEM_TYPE="x86_64"
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
      TARGET="linux-x86_64"
      SYSTEM_TYPE="x86_64"
    fi
    ;;
1)
    TARGET="android-armeabi-v7a"
    SYSTEM_TYPE="armeabi-v7a"
    ;;
2)
    TARGET="android-arm64-v8a"
    SYSTEM_TYPE="arm64-v8a"
    ;;
3)
    TARGET="android-x86_64"
    SYSTEM_TYPE="x86_64"
    ;;
4)
    TARGET="android-x86"
    SYSTEM_TYPE="x86"
    ;;
*)
    TARGET="osx-x86_64"
    SYSTEM_TYPE="x86_64"
    ;;
esac

echo "===== build target is $TARGET, build type is $BUILD_TYPE"

android_target_tag="android"
if [[ $TARGET == *$android_target_tag* ]]; then
  if [ "$ANDROID_NDK_HOME" = "" ]; then
    echo "ERROR: Please set ANDROID_NDK_HOME environment"
    exit
  fi
  echo "===== ANDROID_NDK_HOME=$ANDROID_NDK_HOME"
  ANDROID_TOOLCHAIN=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake
fi


# create build dir if not exists
if [ ! -d build ]; then
    mkdir -p build
fi
cd build

BUILD_DIR="$TARGET-$BUILD_TYPE"
  echo "===== ------ Begin to build xperf ------"
  if [ ! -d $BUILD_DIR ]; then
      mkdir -p $BUILD_DIR
  fi
  cd $BUILD_DIR

  # compile & install
  if [ "$TARGET" = "android-armeabi-v7a" ]; then
      echo "===== cmake target: android-armeabi-v7a"
       cmake   -DICP_TARGET_OS=android \
               -DANDROID_ABI=armeabi-v7a \
               -DANDROID_PLATFORM=android-23 \
               -DANDROID_ARM_NEON=ON \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN \
               -DENABLE_SNPE=$ENABLE_SNPE \
               -DENABLE_QNN=$ENABLE_QNN \

               ../..

  elif [ "$TARGET" = "android-arm64-v8a" ]; then
      echo "===== cmake target: android-arm64-v8a"
       cmake   -DICP_TARGET_OS=android \
               -DANDROID_ABI=arm64-v8a \
               -DANDROID_PLATFORM=android-23 \
               -DANDROID_ARM_NEON=ON \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN \
               -DENABLE_SNPE=$ENABLE_SNPE \
               -DENABLE_QNN=$ENABLE_QNN \
               ../..

  elif [ "$TARGET" = "android-x86_64" ]; then
      echo "===== cmake target: android-x86_64"
      cmake    -DICP_TARGET_OS=android \
               -DANDROID_ABI=x86_64 \
               -DANDROID_PLATFORM=android-23 \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN \
               ../..

  elif [ "$TARGET" = "android-x86" ]; then
      echo "===== cmake target: android-x86"
      cmake    -DICP_TARGET_OS=android \
               -DANDROID_ABI=x86 \
               -DANDROID_PLATFORM=android-23 \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               -DCMAKE_TOOLCHAIN_FILE=$ANDROID_TOOLCHAIN \
               ../..

  elif [ "$TARGET" = "osx-x86_64" ]; then
      echo "===== cmake target: osx-x86_64"
       cmake   -DICP_TARGET_OS=osx \
               -DICP_TARGET_ARCH=x86_64 \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               ../..
  elif [ "$TARGET" = "linux-x86_64" ]; then
      echo "===== cmake target: linux-x86_64"
       # linux support qnn inference engine
       cmake   -DICP_TARGET_OS=linux \
               -DICP_TARGET_ARCH=x86_64 \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               -DENABLE_QNN=$ENABLE_QNN \
               ../..
  fi

  make -j 12
  echo "===== make xperf finished!"
  cd -