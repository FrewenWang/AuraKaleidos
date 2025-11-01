#!/bin/bash

# Default values
ANDROID_ABI="arm64-v8a"
AURA_ENABLE_ARM82="ON"
AURA_LIB_TYPE="static"
AURA_BUILD_TYPE="Release"
AURA_ENABLE_ASAN="OFF"
AURA_ENABLE_HWASAN="OFF"
AURA_ENABLE_OPENCL="OFF"
AURA_ENABLE_HEXAGON="OFF"
AURA_ENABLE_NN="OFF"
AURA_BUILD_UNIT_TEST="OFF"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --android_abi <value>          Set the Android ABI, default is arm64-v8a
  --enable_arm82 <value>         Enable ARM82 support, default is ON
  --lib_type <value>             Set the library type, default is static
  --build_type <value>           Set the build type, default is Release
  --enable_asan <value>          Enable AddressSanitizer, default is OFF
  --enable_hwasan <value>        Enable HWAddressSanitizer, default is OFF
  --enable_opencl <value>        Enable OpenCL, default is OFF
  --enable_hexagon <value>       Enable Hexagon support, default is OFF
  --enable_nn <value>            Enable Neural Network support, default is OFF
  --build_unit_test <value>      Build unit tests, default is OFF
  -h, --help                     Show this help message and exit

e.g.
    # Build with hwasan and build unit tests
    $0 --enable_hwasan ON --build_unit_test ON

    # Build with OpenCL and static library
    $0 --enable_opencl ON --build_type Debug
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --android_abi)
                ANDROID_ABI="$2"
                shift # past argument
                shift # past value
                ;;
            --enable_arm82)
                AURA_ENABLE_ARM82="$2"
                shift # past argument
                shift # past value
                ;;
            --lib_type)
                AURA_LIB_TYPE="$2"
                shift # past argument
                shift # past value
                ;;
            --build_type)
                AURA_BUILD_TYPE="$2"
                shift # past argument
                shift # past value
                ;;
            --enable_asan)
                AURA_ENABLE_ASAN="$2"
                shift # past argument
                shift # past value
                ;;
            --enable_hwasan)
                AURA_ENABLE_HWASAN="$2"
                shift # past argument
                shift # past value
                ;;
            --enable_opencl)
                AURA_ENABLE_OPENCL="$2"
                shift # past argument
                shift # past value
                ;;
            --enable_hexagon)
                AURA_ENABLE_HEXAGON="$2"
                shift # past argument
                shift # past value
                ;;
            --enable_nn)
                AURA_ENABLE_NN="$2"
                shift # past argument
                shift # past value
                ;;
            --build_unit_test)
                AURA_BUILD_UNIT_TEST="$2"
                shift # past argument
                shift # past value
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Parse command-line arguments
parse_args "$@"

# Check required variables
if [ -z "$OPENCV_ROOT" ]; then
    echo "OPENCV_ROOT is not set. Please provide it via --OPENCV_ROOT."
    exit 1
fi

if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "ANDROID_NDK_ROOT is not set. Please provide it via --ANDROID_NDK_ROOT."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_PATH="$SCRIPT_DIR"/../../build_all/build_android

echo "Build path: $BUILD_PATH"

if [ ! -d "$BUILD_PATH" ]; then
    mkdir -p "$BUILD_PATH"
fi

cd "$BUILD_PATH"

# Echo configuration for verification
echo "Using the following configuration:"
echo "ANDROID_ABI: $ANDROID_ABI"
echo "AURA_ENABLE_ARM82: $AURA_ENABLE_ARM82"
echo "AURA_LIB_TYPE: $AURA_LIB_TYPE"
echo "AURA_BUILD_TYPE: $AURA_BUILD_TYPE"
echo "AURA_ENABLE_ASAN: $AURA_ENABLE_ASAN"
echo "AURA_ENABLE_HWASAN: $AURA_ENABLE_HWASAN"
echo "AURA_ENABLE_OPENCL: $AURA_ENABLE_OPENCL"
echo "AURA_ENABLE_HEXAGON: $AURA_ENABLE_HEXAGON"
echo "AURA_ENABLE_NN: $AURA_ENABLE_NN"
echo "AURA_BUILD_UNIT_TEST: $AURA_BUILD_UNIT_TEST"

echo "ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
echo "OpenCV_DIR: $OPENCV_ROOT"

# Build
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake          \
    -DANDROID_ABI=$ANDROID_ABI                                                            \
    -DANDROID_STL=c++_shared                                                              \
    -DANDROID_PLATFORM=android-23                                                         \
    -DAURA_ENABLE_ARM82=$AURA_ENABLE_ARM82                                                \
    -DAURA_LIB_TYPE=$AURA_LIB_TYPE                                                        \
    -DAURA_BUILD_TYPE=$AURA_BUILD_TYPE                                                    \
    -DAURA_ENABLE_ASAN=$AURA_ENABLE_ASAN                                                  \
    -DAURA_ENABLE_HWASAN=$AURA_ENABLE_HWASAN                                              \
    -DAURA_ENABLE_OPENCL=$AURA_ENABLE_OPENCL                                              \
    -DAURA_ENABLE_HEXAGON=$AURA_ENABLE_HEXAGON                                            \
    -DAURA_ENABLE_NN=$AURA_ENABLE_NN                                                      \
    -DAURA_OPENCL_VERSION=200                                                             \
    -DAURA_BUILD_UNIT_TEST=$AURA_BUILD_UNIT_TEST                                          \
    -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake                                             \
    ../..

make install -j8
