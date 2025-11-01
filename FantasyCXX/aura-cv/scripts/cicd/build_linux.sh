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

build_linux_config() {
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
    echo "******************************** linux new type ********************************"

    echo "aura_version: $aura_version_major.$aura_version_minor.$aura_version_patch"

    local enable_shared_lib="OFF"
    if [[ "$lib" == "share" ]]; then
        enable_shared_lib="ON"
    fi

    echo "work_path: $work_path"
    cd "$work_path"

    build_dir="build_linux/$arch/$lib/$type"

    mkdir -p "$build_dir"
    cd "$build_dir"

    echo "build_dir: $build_dir"

    if [[ -d "install" ]]; then
        rm -rf "install"
    fi

    cmake   -DAURA_LINUX_ABI="$arch" \
            -DAURA_SHARED_LIBRARY="$enable_shared_lib" \
            -DAURA_BUILD_TYPE="$type" \
            -DAURA_VERSION="$aura_version_major.$aura_version_minor.$aura_version_patch" \
            -DAURA_BUILD_UNIT_TEST=ON                  \
            -DOpenCV_DIR=$OPENCV_ROOT/linux-gcc7.5.0-x64/lib/cmake/opencv4  \
            ../../../..
    Error $? "${LOG_TAG} cmake failed"

    cmake --build . --target install -- -j 64
    Error $? "${LOG_TAG} build failed"

    # Use lsb_release command to fetch Ubuntu version information
    ubuntu_version_info=$(lsb_release -a 2>/dev/null)

    if [ -n "$ubuntu_version_info" ]; then
        # Extract the Release field from the version information
        release=$(echo "$ubuntu_version_info" | grep 'Release:' | awk '{print $2}')

        # Extract the major version number
        ubuntu_major_version=$(echo "$release" | cut -d'.' -f1)

        # Print the major version number
        echo "Ubuntu major version: $ubuntu_major_version"
    else
        Error 1 "${LOG_TAG} unable to get ubuntu version information"
    fi

    # Parse the vesrison of glibc
    glibc_version=$(ldd --version | head -n 1 | grep -oP '\d+\.\d+' | head -n 1)
    if [ -n "$glibc_version" ]; then
        echo "glibc version: $glibc_version"
    else
        Error 1 "${LOG_TAG} unable to get glibc version information"
    fi

    cd ./install

    for dir in */; do
        dir=${dir%/}

        zip_sub_name="aura_${aura_version_major}.${aura_version_minor}.${aura_version_patch}_ubuntu${ubuntu_major_version}-glibc${glibc_version}_${arch}_${lib}_${type}"
        zip_name="$zip_sub_name.zip"

        mv "$dir" "$zip_sub_name"
        zip -r "$zip_name" "$zip_sub_name/" > /dev/null
        Error $? "${LOG_TAG} zip failed"
        echo "******packed successfully for name: $zip_name"

        mkdir -p "$release_path/linux/"

        [ ! -d "$release_path/linux/" ] && Error 1 "Failed to create directory $release_path/linux/"

        cp "$zip_name" "$release_path/linux/"
        Error $? "${LOG_TAG} copy failed: $zip_name to $release_path/linux/"
        echo "copy $zip_name to $release_path/linux/"

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

build_linux() {
    [ $# -lt 5 ] && Error 1 "param < 5"

    local work_path=$1
    local release_path=$2

    local aura_version_major=$3
    local aura_version_minor=$4
    local aura_version_patch=$5

    local archs=("x64")
    local libs=("static" "share")
    local types=("release")

    # delete old build and release
    [ -e "${work_path}/build_linux" ] && rm -rf "${work_path}/build_linux"
    [ -e "$release_path/linux/" ] && rm -rf "$release_path/linux/"

    for arch in ${archs[@]}; do
        for lib in ${libs[@]}; do
            for type in ${types[@]}; do
                build_linux_config "$work_path" "$release_path" "$arch" "$lib" "$type" "$aura_version_major" "$aura_version_minor" "$aura_version_patch"
            done
        done
    done
}

# usages:
# ./build_linux.sh <work_path> <release_path> <aura_version_major> <aura_version_minor> <aura_version_patch>
#   <work_path> : absulute path of aura project
#   <release_path> : absulute path of aura release
# such as: bash ./scripts/cicd/build_linux.sh $(pwd) $(pwd)/release 2 0 0
build_linux "$@"