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
ODB_BASE=$(cd "${SCRIPT_DIR}"; pwd)
case "$HOST_OS" in
	Linux)
    ODB_HOST="$ODB_BASE/bin"
    ODB_HOST_TARGET="$ODB_BASE/Linux"
		;;
	Darwin)
    ODB_HOST="$ODB_BASE/bin"
    ODB_HOST_TARGET="$ODB_BASE/Darwin"
		;;
	*)
    ODB_HOST=$ODB_BASE
		;;
esac

PATH=$ODB_HOST:$ODB_HOST_TARGET:$PATH

chmod a+x -R $ODB_HOST/*

export ODB_HOST PATH ODB_HOST_TARGET

echo ODB_HOST=$ODB_HOST
echo ODB_HOST_TARGET=$ODB_HOST_TARGET

#Do not edit past this line
