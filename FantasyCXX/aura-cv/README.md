# aura

aura2.0

## Android arm64-v8a debug with asan cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="arm64-v8a"                                                             \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_ENABLE_ARM82=ON                                                                \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_TYPE=debug                                                               \
    -DAURA_ENABLE_ASAN=ON                                                                 \
    -DAURA_ENABLE_OPENCL=ON                                                               \
    -DAURA_ENABLE_HEXAGON=ON                                                              \
    -DAURA_ENABLE_NN=ON                                                                   \
    -DAURA_OPENCL_VERSION=200                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake ..
```

## Android arm64-v8a debug with hwasan cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="arm64-v8a"                                                             \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_ENABLE_ARM82=ON                                                                \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_TYPE=debug                                                               \
    -DAURA_ENABLE_HWASAN=ON                                                               \
    -DAURA_ENABLE_OPENCL=ON                                                               \
    -DAURA_ENABLE_HEXAGON=ON                                                              \
    -DAURA_ENABLE_NN=ON                                                                   \
    -DAURA_OPENCL_VERSION=200                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake ..
```

## Android arm64-v8a release cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="arm64-v8a"                                                             \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_ENABLE_ARM82=ON                                                                \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_ENABLE_OPENCL=ON                                                               \
    -DAURA_ENABLE_HEXAGON=ON                                                              \
    -DAURA_ENABLE_NN=ON                                                                   \
    -DAURA_OPENCL_VERSION=200                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake                                             \
    ..
```

## Android arm64-v8a release nnlite cmd

```bash
# nnlite will forcibly turn on AURA_ENABLE_ARM82 and AURA_ENABLE_NN, and turn off AURA_ENABLE_HEXAGON, AURA_ENABLE_OPENCL, and AURA_BUILD_UNIT_TEST.

cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="arm64-v8a"                                                             \
    -DANDROID_PLATFORM=android-23                                                         \
    -DANDROID_STL=c++_shared                                                              \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_ENABLE_ARM82=ON                                                                \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_ENABLE_NN=ON                                                                   \
    -DAURA_ENABLE_NN_LITE=ON                                                              \
    -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake                                             \
    ..
```

## Android armeabi-v7a release cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="armeabi-v7a"                                                           \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_ENABLE_ARM82=OFF                                                               \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_ENABLE_OPENCL=ON                                                               \
    -DAURA_ENABLE_HEXAGON=ON                                                              \
    -DAURA_ENABLE_NN=ON                                                                   \
    -DAURA_OPENCL_VERSION=200                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/armeabi-v7a/cmake                                           \
    ..
```

## Host compiler linux x64 cmd

```bash
cmake                                                                                     \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_LINUX_ABI=x64                                                                  \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/linux-gcc7.5.0-x64/lib/cmake/opencv4                        \
    ..
```

## Host compiler linux x86 cmd

```bash
cmake                                                                                     \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_LINUX_ABI=x86                                                                  \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/linux-gcc7.5.0-x86/lib/cmake/opencv4                        \
    ..
```

## Host compiler windows x64 cmd with vs2019

```bash
cmake                                                                                     \
    -A x64 -G "Visual Studio 16 2019"                                                     \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/windows-x64                                                 \
    ..
```

## Host compiler windows x86 cmd with vs2019

```bash
cmake                                                                                     \
    -A win32 -G "Visual Studio 16 2019"                                                   \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/windows-x86                                                 \
    ..
```

## hexaogn compile sdk requirement

hexagon sdk4.2: v65/v66/v68
hexagon sdk5.1: v65/v66/v68/v69
hexagon sdk5.3: v65/v66/v68/v69/v73/v75
hexagon sdk6.0: v65/v66/v68/v69/v73/v75/v79
hexagon sdk6.2: v65/v66/v68/v69/v73/v75/v79/v81

hexagon comiple must -DAURA_HEXAGON_ARCH

## Hexagon compiler release cmd, v73/v75 must use hexaogn sdk5.3

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/hexagon_toolchain.cmake                               \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_HEXAGON_ARCH=v75                                                               \
    ..
```

## Hexagon compiler with hexagon-side unit_test cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/hexagon_toolchain.cmake                               \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=OFF                                                             \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DAURA_HEXAGON_ARCH=v68                                                               \
    ..
```

## Qnn Udo Host compiler linux x64 release cmd

```bash
cmake                                                                                     \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=OFF                                                             \
    -DAURA_LINUX_ABI=x64                                                                  \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_LINUX_COMPILER=clang                                                           \
    -DAURA_ENABLE_NN=ON                                                                   \
    ..
```

## Qnn Udo Android arm64-v8a release cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="arm64-v8a"                                                             \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_ENABLE_ARM82=ON                                                                \
    -DAURA_SHARED_LIBRARY=OFF                                                             \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_ENABLE_NN=ON                                                                   \
    ..
```

## Qnn Udo Hexagon compiler with hexagon-side unit_test cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/hexagon_toolchain.cmake                               \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_SHARED_LIBRARY=OFF                                                             \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DAURA_HEXAGON_ARCH=v68                                                               \
    ..
```

## xtensa compile requirement

```bash
export XTENSA_SDK_PATH=$YOUR_XTENSA_PATH
xtensa comiple must -DAURA_XTENSA_CORE
```

## Cadence compiler with xtensa-side cmd

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/xtensa_toolchain.cmake                                \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_BUILD_XTENSA=ON                                                                \
    -DAURA_XTENSA_CORE=vq8                                                                \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_SHARED_LIBRARY=OFF                                                             \
    ..
```

## Cadence compiler with xplorer-side cmd for profile on Xplorer

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=./cmake/xtensa_toolchain.cmake                                 \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_BUILD_XPLORER=ON                                                               \
    -DAURA_XTENSA_CORE=vq8                                                                \
    -DAURA_SHARED_LIBRARY=OFF                                                             \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DAURA_XTENSA_LIB_DIR=${YOUR_AURA_XTENSA_LIB_DIR}                                     \
    ..
```

## Android arm64-v8a release cmd with xtensa

```bash
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI="arm64-v8a"                                                             \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_ARM_NEON=ON                                                                 \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_VERSION=2.1.1                                                                  \
    -DAURA_ENABLE_ARM82=ON                                                                \
    -DAURA_SHARED_LIBRARY=ON                                                              \
    -DAURA_ENABLE_OPENCL=ON                                                               \
    -DAURA_ENABLE_XTENSA=ON                                                               \
    -DAURA_OPENCL_VERSION=200                                                             \
    -DAURA_BUILD_TYPE=release                                                             \
    -DAURA_BUILD_UNIT_TEST=ON                                                             \
    -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake                                             \
    ..
```
