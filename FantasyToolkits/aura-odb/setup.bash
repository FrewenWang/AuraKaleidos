#!/bin/bash

# shellcheck disable=SC2086
if [ -z $ODB_HOME ]; then
  ODB_HOME="$PWD/odb-env.sh"
  echo "Setup ODB_HOME=$ODB_HOME"
else
  echo "ODB_HOME=$ODB_HOME"
fi

# HOST_OS=$(uname -s)
# case "$HOST_OS" in
# 	Linux)
#     echo "ODB_HOME=$ODB_HOME" >> ~/.bashrc
#     echo "source \$ODB_HOME"  >> ~/.bashrc
#     source ~/.bashrc
# 		;;
# 	Darwin)
# 		;;

# 	*)
# 		;;
# esac