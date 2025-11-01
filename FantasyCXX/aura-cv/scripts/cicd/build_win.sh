#!/bin/bash

# helper function
Error()
{
    if [ $# -eq 2 ] && [ $1 != 0 ]; then
        echo -e "\033[41;37m[aura builder]: $2\033[0m"
        exit 1
    fi
}

build_win_docker()
{
    [ $# -lt 5 ] && Error 1 "param < 5"

    local work_path=$1
    local release_path=$2

    local aura_version_major=$3
    local aura_version_minor=$4
    local aura_version_patch=$5


    local mount_path=/aura2.0_source
    local mount_release_path="$mount_path/$release_path/windows"

    echo "run docker"

    docker run --rm --user $UID -v ${work_path}:${mount_path} msvc_aura2.0 /bin/bash -c "${mount_path}/scripts/cicd/build_win_docker.sh ${mount_path} ${mount_release_path} ${aura_version_major} ${aura_version_minor} ${aura_version_patch}"
    Error $? "docker build win64"
}

# usages:
# ./build_win.sh <work_path> <release_path> <aura_version_major> <aura_version_minor> <aura_version_patch>
#   <work_path> : absulute path of aura project
#   <release_path> : relative path of aura release
# such as: bash ./scripts/cicd/build_win.sh $(pwd) release 2 0 0
build_win_docker "$@"