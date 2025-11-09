#!/bin/bash
# This script sets environment variables required to use this version of Xiaomi Phone Software Development Platform
# from the command line. To use the script, you have to "source" it into your shell, i.e.:
#   source midb-env.sh
# if source command is not available use "." (dot) command instead
#
# shellcheck disable=SC2128
test "$BASH_SOURCE" = "" && echo "This script can be sourced only from bash" && return
SCRIPT_SOURCE=$BASH_SOURCE
test "$SCRIPT_SOURCE" = "$0" && echo "Script is being run, should be sourced" && exit 1

HOST_OS=$(uname -s)
SCRIPT_DIR=$(dirname "${SCRIPT_SOURCE}")
MIDB_BASE=$(cd "${SCRIPT_DIR}" || exit; pwd)
case "$HOST_OS" in
	Linux)
    MIDB_HOST="$MIDB_BASE/bin"
    MIDB_HOST_TARGET="$MIDB_BASE/Linux"
		;;
	Darwin)
    MIDB_HOST="$MIDB_BASE/bin"
    MIDB_HOST_TARGET="$MIDB_BASE/Darwin"
		;;
	*)
    MIDB_HOST=$MIDB_BASE
		;;
esac

PATH=$MIDB_HOST:$MIDB_HOST_TARGET:$PATH

chmod a+x -R $MIDB_HOST/*

export MIDB_HOST PATH MIDB_HOST_TARGET

echo MIDB_HOST=$MIDB_HOST
echo MIDB_HOST_TARGET=$MIDB_HOST_TARGET

#Do not edit past this line
