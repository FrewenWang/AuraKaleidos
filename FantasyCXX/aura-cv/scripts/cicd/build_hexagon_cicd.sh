#!/bin/bash

# helper function
Error()
{
    if [ $# -eq 2 ] && [ $1 != 0 ]; then
        echo -e "\033[41;37m[aura builder]: $2\033[0m"
        exit 1
    fi
}

LOG_TAG="[aura2.0 build]"

hexagon_release_path=""

build_hexagon_config() {
    # check params
    [ $# -lt 8 ] && Error 1 "param < 8"

    local work_path=$1
    local release_path=$2
    local arch=$3
    local lib=$4
    local type=$5
    local aura_version_major=$6
    local aura_version_minor=$7
    local aura_version_patch=$8

    echo
    echo "******************************** hexagon new type ********************************"

    echo "aura_version: $aura_version_major.$aura_version_minor.$aura_version_patch"

    local enable_shared_lib="OFF"
    if [[ "$lib" == "share" ]]; then
        enable_shared_lib="ON"
    fi

    echo "work_path: $work_path"
    cd "$work_path"

    build_dir="build_hexagon/$arch/$lib/$type/"

    mkdir -p "$build_dir"

    if [ ! -d "$build_dir" ]; then
        echo "Error: Directory $build_dir does not exist."
        exit 1
    fi

    cd "$build_dir"
    echo "build_dir: $build_dir"

    if [[ -d "install" ]]; then
        rm -rf "install"
    fi

    cmake -DCMAKE_TOOLCHAIN_FILE=../../../../cmake/hexagon_toolchain.cmake \
          -DAURA_HEXAGON_ARCH="$arch" \
          -DAURA_SHARED_LIBRARY="$enable_shared_lib" \
          -DAURA_BUILD_TYPE="$type" \
          -DAURA_ENABLE_NN=ON \
          -DAURA_VERSION="$aura_version_major.$aura_version_minor.$aura_version_patch" \
          ../../../..
    Error $? "${LOG_TAG} cmake failed"

    cmake --build . --target install -- -j 64
    Error $? "${LOG_TAG} build failed"

    cd ./install

    for dir in */; do
        dir=${dir%/}

        zip_sub_name="aura_${aura_version_major}.${aura_version_minor}.${aura_version_patch}_hexagon_${arch}_${lib}_${type}"
        zip_name="$zip_sub_name.zip"

        mv "$dir" "$zip_sub_name"
        zip -r "$zip_name" "$zip_sub_name/" > /dev/null
        Error $? "${LOG_TAG} zip failed"
        echo "******packed successfully for name: $zip_name"

        cp "$zip_name" "$hexagon_release_path"
        Error $? "${LOG_TAG} copy failed: $zip_name to $hexagon_release_path"
        echo "copy $zip_name to $hexagon_release_path"

        # then delete the zip file in current dir
        [ -e "$zip_name" ] && rm "$zip_name"
        Error $? "${LOG_TAG} delete failed: $zip_name"
    done

    cd "$work_path/$build_dir"
    Error $? "${LOG_TAG} cd failed: $work_path/$build_dir"

    # Delete the build_hexagon/**/src/ folder to reduce the total build folder size
    [ -e "src/" ] && rm -rf "src/"
    Error $? "${LOG_TAG} delete failed: src/ in path: $build_dir"

    cd "$work_path"
}

build_hexagon() {
    # check params
    [ $# -lt 5 ] && Error 1 "param < 5"

    local work_path=$1
    local release_path=$2

    local aura_version_major=$3
    local aura_version_minor=$4
    local aura_version_patch=$5

    if [ -z "$HEXAGON_SDK_PATH" ]; then
        echo "HEXAGON_SDK_PATH is not set"
        exit 1
    else
        echo "HEXAGON_SDK_PATH is set to $HEXAGON_SDK_PATH"
    fi

    # delete old build and release
    [ -e "${work_path}/build_hexagon" ] && rm -rf "${work_path}/build_hexagon"

    hexagon_release_path="$release_path/android/"
    echo "hexagon_release_path: $hexagon_release_path"
    mkdir -p "$hexagon_release_path"
    [ ! -d "$hexagon_release_path" ] && Error 1 "Failed to create directory $hexagon_release_path"

    local archs=("v68" "v69" "v73" "v75" "v79" "v81")
    local libs=("static" "share")
    local types=("release")

    for arch in "${archs[@]}"; do
        for lib in "${libs[@]}"; do
            build_hexagon_config $1 $2 "$arch" "$lib" "$types" "$aura_version_major" "$aura_version_minor" "$aura_version_patch"
        done
    done

    cd "$work_path"
}

# set build env
source /home/mi-aura/workspace/aura2.0_build_env/set_env.sh

build_hexagon "$1" "$2" "$3" "$4" "$5"
