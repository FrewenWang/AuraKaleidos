#!/bin/bash

# Default values
HEXAGON_ARCH="v69"
LIB_TYPE="share"
BUILD_TYPE="release"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --hexagon_arch <value>        Set the Hexagon architecture(v68/69/73/75), default is v69
  --lib_type <value>            Set the library type, default is shared
  --build_type <value>          Set the build type, default is release
  -h, --help                    Show this help message and exit

e.g.
    # Build with specific Hexagon architecture
    $0 --hexagon_arch v73 --lib_type static --build_type debug
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --hexagon_arch)
                HEXAGON_ARCH="$2"
                shift # past argument
                shift # past value
                ;;
            --lib_type)
                LIB_TYPE="$2"
                shift # past argument
                shift # past value
                ;;
            --build_type)
                BUILD_TYPE="$2"
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

# Check required variables (if any)
if [ -z $HEXAGON_SDK_PATH ]; then
    echo "Error: HEXAGON_SDK_PATH is not set, please set it to the Hexagon SDK path"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_PATH="$SCRIPT_DIR"/../../build_all/build_hexagon

echo "Build path: $BUILD_PATH"

if [ ! -d "$BUILD_PATH" ]; then
    mkdir -p "$BUILD_PATH"
fi

cd "$BUILD_PATH"

# Echo configuration for verification
echo "Using the following configuration:"
echo "HEXAGON_ARCH: $HEXAGON_ARCH"
echo "LIB_TYPE: $LIB_TYPE"
echo "BUILD_TYPE: $BUILD_TYPE"

# CMake configuration
cmake                                                                                     \
    -DCMAKE_TOOLCHAIN_FILE="$SCRIPT_DIR/../../cmake/hexagon_toolchain.cmake"              \
    -DAURA_HEXAGON_ARCH="$HEXAGON_ARCH"                                                   \
    -DAURA_LIB_TYPE="$LIB_TYPE"                                                           \
    -DAURA_BUILD_TYPE="$BUILD_TYPE"                                                       \
    ../..

make install -j8
