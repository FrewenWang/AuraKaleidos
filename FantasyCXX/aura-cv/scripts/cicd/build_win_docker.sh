#!/bin/bash

# helper function
Error()
{
    if [ $# -eq 2 ] && [ $1 != 0 ]; then
        echo -e "\033[41;37m[aura builder]: $2\033[0m"
        exit 1
    fi
}

build_win_config() {
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
    echo "******************************** win new type ********************************"

    echo "aura_version: $aura_version_major.$aura_version_minor.$aura_version_patch"

    local enable_shared_lib="OFF"
    if [[ "$lib" == "share" ]]; then
        enable_shared_lib="ON"
    fi

    echo "work_path: $work_path"
    cd "$work_path"

    build_dir="build_win/$arch/$lib/$type"

    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "build_dir: $build_dir"

    if [[ -d "install" ]]; then
        rm -rf "install"
    fi

    cmake   -GNinja -DCMAKE_TOOLCHAIN_FILE=/opt/cross-tools/windows-amd64.cmake \
            "x64" \
            -DAURA_SHARED_LIBRARY="$enable_shared_lib" \
            -DAURA_BUILD_TYPE="$type" \
            -DAURA_VERSION="$aura_version_major.$aura_version_minor.$aura_version_patch" \
            ../../../..

    Error $? "exec cmake failed"

    # exec build
    ninja -j64 install
    Error $? "exec make failed"

    cd ./install

    for dir in */; do
        dir=${dir%/}

        zip_sub_name="aura_${aura_version_major}.${aura_version_minor}.${aura_version_patch}_windows_${arch}_vc19_${lib}_${type}"
        zip_name="$zip_sub_name.zip"

        mv "$dir" "$zip_sub_name"
        zip -r "$zip_name" "$zip_sub_name/" > /dev/null
        Error $? "${LOG_TAG} zip failed"
        echo "******packed successfully for name: $zip_name"

        mkdir -p "$release_path"

        [ ! -d "$release_path" ] && Error 1 "Failed to create directory $release_path"

        cp "$zip_name" "$release_path"
        Error $? "${LOG_TAG} copy failed: $zip_name to $release_path"
        echo "copy $zip_name to $release_path"

        # then delete the zip file in current dir
        [ -e "$zip_name" ] && rm "$zip_name"
        Error $? "${LOG_TAG} delete failed: $zip_name"
    done

    cd "$work_path/$build_dir"
    Error $? "${LOG_TAG} cd failed: $work_path/$build_dir"

    # Delete the build_linux/**/src/ folder to reduce the total build folder size
    [ -e "src/" ] && rm -rf "src/"
    Error $? "${LOG_TAG} delete failed: src/ in path: $build_dir"

    cd "$work_path"
}

build_win() {
    [ $# -lt 5 ] && Error 1 "param < 5"

    local work_path=$1
    local release_path=$2

    local aura_version_major=$3
    local aura_version_minor=$4
    local aura_version_patch=$5

    local archs=("x64")
    local libs=("static" "share")
    local types=("release" "debug")

    # delete old build and release
    [ -e "${work_path}/build_win" ] && rm -rf "${work_path}/build_win"
    [ -e "$release_path" ] && rm -rf "$release_path"

    for arch in ${archs[@]}; do
        for lib in ${libs[@]}; do
            for type in ${types[@]}; do
                build_win_config "$work_path" "$release_path" "$arch" "$lib" "$type" "$aura_version_major" "$aura_version_minor" "$aura_version_patch"
            done
        done
    done
}

build_win "$@"