#!/bin/bash
# This script sets environment variables required to use this version of QNX Software Development Platform
# from the command line. To use the script, you have to "source" it into your shell, i.e.:
#   source qnxsdp-env.sh
# if source command is not available use "." (dot) command instead
#
test "$BASH_SOURCE" = "" && echo "This script can be sourced only from bash" && return
SCRIPT_SOURCE=$BASH_SOURCE
test "$SCRIPT_SOURCE" = "$0" && echo "Script is being run, should be sourced" && exit 1

HOST_OS=$(uname -s)
SCRIPT_DIR=$(dirname "${SCRIPT_SOURCE}")
QDB_BASE=$(cd "${SCRIPT_DIR}"; pwd)
case "$HOST_OS" in
	Linux)
    QDB_HOST="$QDB_BASE/bin"
    QDB_HOST_TARGET="$QDB_BASE/Linux"
		;;
	Darwin)
    QDB_HOST="$QDB_BASE/bin"
    QDB_HOST_TARGET="$QDB_BASE/Darwin"
		;;
	*)
    QDB_HOST=$QDB_BASE
		;;
esac

PATH=$QDB_HOST:$QDB_HOST_TARGET:$PATH

chmod a+x -R $QDB_HOST/*

export QDB_HOST PATH QDB_HOST_TARGET

echo QDB_HOST=$QDB_HOST
echo QDB_HOST_TARGET=$QDB_HOST_TARGET

#Do not edit past this line
