#!/bin/bash

if [ -z $QDB_HOME ]; then
  QDB_HOME="$PWD/qdb-env.sh"
  echo "QDB_HOME=$QDB_HOME" >> ~/.bashrc
  echo "source \$QDB_HOME"  >> ~/.bashrc
  source ~/.bashrc
else
  echo "QDB_HOME=$QDB_HOME" >> ~/.bashrc
fi