#!/bin/bash


if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <binary_file> <cash_addr>"
    exit 1
fi

BINARY=$1
ADDR=$2


addr2line -e "$BINARY" -f -C -p "$ADDR"