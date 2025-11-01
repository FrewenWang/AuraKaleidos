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

build_android_config() {
    # check params
    [ $# -lt 12 ] && Error 1 "param < 12"

    local work_path=$1
    local release_path=$2

    local platform=$3
    local arch=$4
    local lib=$5
    local type=$6
    local asan=$7
    local aura_version_major=$8
    local aura_version_minor=$9
    local aura_version_patch=${10}

    local ndk_version=${11}
    local android_ndk_root=${12}

    # check 'realease' only corresponding 'non_asan'
    # 1. release: non_asan
    # 2. debug: asan/hwasan
    if [ "$type" == "release" ]; then
        if [ "$asan" == "hwasan" ] || [ "$asan" == "asan" ]; then
            return;
        fi
    elif [ "$type" == "debug" ]; then
        if [ "$asan" == "non_asan" ]; then
            return;
        fi
    fi

    echo
    echo "******************************** android new type ********************************"

    echo "aura_version: $aura_version_major.$aura_version_minor.$aura_version_patch"
    echo "ndk_version: $ndk_version"
    echo "android_ndk_root: $android_ndk_root"

    local qcom="ON"
    local nnlite="OFF"
    local device_name="android"
    local nn_name=""
    if [[ "$platform" == "nnlite" ]]; then
        nnlite="ON"
        nn_name="_nnlite"
    fi

    local cmake_hwasan="OFF"
    local cmake_asan="OFF"
    local asan_name=""
    if [[ "$asan" == "hwasan" ]]; then
        cmake_hwasan="ON"
        cmake_asan="ON"
        asan_name="_hwasan"
    elif [[ "$asan" == "asan" ]]; then
        cmake_asan="ON"
        asan_name="_asan"
    fi

    local enable_shared_lib="OFF"
    if [[ "$lib" == "share" ]]; then
        enable_shared_lib="ON"
    fi

    echo "work_path: $work_path"
    cd "$work_path"

    build_dir="build_android/$platform/$arch/$lib/$type/$asan"

    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "build_dir: $build_dir"

    if [[ -d "install" ]]; then
        rm -rf "install"
    fi

    cmake -DCMAKE_TOOLCHAIN_FILE="$android_ndk_root/build/cmake/android.toolchain.cmake" \
            -DANDROID_PLATFORM=android-23 \
            -DANDROID_ABI="$arch" \
            -DAURA_SHARED_LIBRARY="$enable_shared_lib" \
            -DAURA_BUILD_TYPE="$type" \
            -DANDROID_STL=c++_shared \
            -DAURA_ENABLE_OPENCL=ON \
            -DAURA_OPENCL_VERSION=200 \
            -DAURA_ENABLE_HEXAGON="$qcom" \
            -DAURA_ENABLE_NN=ON \
            -DAURA_ENABLE_NN_LITE="$nnlite" \
            -DAURA_ENABLE_ASAN="$cmake_asan" \
            -DAURA_ENABLE_HWASAN="$cmake_hwasan" \
            -DAURA_VERSION="$aura_version_major.$aura_version_minor.$aura_version_patch" \
            -DAURA_BUILD_UNIT_TEST=ON                  \
            -DOpenCV_DIR=$OPENCV_ROOT/arm64-v8a/cmake  \
            ../../../../../..
    Error $? "${LOG_TAG} cmake failed"

    cmake --build . --target install -- -j 64
    Error $? "${LOG_TAG} build failed"

    cd ./install

    for dir in */; do
        dir=${dir%/}

        zip_sub_name="aura_${aura_version_major}.${aura_version_minor}.${aura_version_patch}${nn_name}_${device_name}_${arch}_${ndk_version}_${lib}_${type}${asan_name}"
        zip_name="$zip_sub_name.zip"

        mv "$dir" "$zip_sub_name"
        zip -r "$zip_name" "$zip_sub_name/" > /dev/null
        Error $? "${LOG_TAG} zip failed"
        echo "******packed successfully for name: $zip_name"

        mkdir -p "$release_path/android/"

        [ ! -d "$release_path/android/" ] && Error 1 "Failed to create directory $release_path/android/"

        cp "$zip_name" "$release_path/android/"
        Error $? "${LOG_TAG} copy failed: $zip_name to $release_path/android/"
        echo "copy $zip_name to $release_path/android/"

        # then delete the zip file in current dir
        [ -e "$zip_name" ] && rm "$zip_name"
        Error $? "${LOG_TAG} delete failed: $zip_name"
    done

    cd "$work_path/$build_dir"
    Error $? "${LOG_TAG} cd failed: $work_path/$build_dir"

    # Delete the build_android/**/src/ folder to reduce the total build folder size
    [ -e "src/" ] && rm -rf "src/"
    Error $? "${LOG_TAG} delete failed: src/ in path: $build_dir"

    cd "$work_path"
}

build_android() {
    # check params
    [ $# -lt 5 ] && Error 1 "param < 5"

    local work_path=$1
    local release_path=$2

    local aura_version_major=$3
    local aura_version_minor=$4
    local aura_version_patch=$5

    local ndk_version=""
    if [ -z "$NDK_PATH" ]; then
        echo "NDK_PATH is not set"
        exit 1
    else
        echo "NDK_PATH is set to $NDK_PATH"
        ndk_version="ndkr26d"

        echo "ndk version: $ndk_version"
    fi

    # delete old build and release
    [ -e "${work_path}/build_android" ] && rm -rf "${work_path}/build_android"
    [ -e "$release_path/android/" ] && rm -rf "$release_path/android/"

    local platforms=("android" "nnlite")
    local archs=("arm64-v8a")
    local libs=("static" "share")
    local asans=("non_asan" "asan" "hwasan")
    local types=("release" "debug")

    for platform in "${platforms[@]}"; do
        for arch in "${archs[@]}"; do
            for lib in "${libs[@]}"; do
                for type in "${types[@]}"; do
                    for asan in "${asans[@]}"; do
                        build_android_config $1 $2 "$platform" "$arch" "$lib" "$type" "$asan" "$aura_version_major" "$aura_version_minor" "$aura_version_patch" "$ndk_version" "$NDK_PATH"
                    done
                done
            done
        done
    done
}

# set build env
source /home/mi-aura/workspace/aura2.0_build_env/set_env.sh

build_android "$1" "$2" "$3" "$4" "$5"
