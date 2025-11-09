#!/bin/bash

if [ -z $DB_HOME ]; then
  MIDB_HOME="$PWD/qdb-env.sh"
  echo "MIDB_HOME=$MIDB_HOME" >> ~/.bashrc
  echo "source \$MIDB_HOME"  >> ~/.bashrc
  source ~/.bashrc
else
  echo "MIDB_HOME=$MIDB_HOME" >> ~/.bashrc
fi