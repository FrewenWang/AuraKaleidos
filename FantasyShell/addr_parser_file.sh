#!/bin/bash

# Usage:
# ./parse_crash_log.sh crash.log ./your_executable

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <crash_log_file> <binary_file>"
    exit 1
fi

LOGFILE=$1
BINARY=$2

echo "Parsing addresses in '$LOGFILE' using '$BINARY'..."
echo "=============================================="

# 从日志中提取所有十六进制地址
ADDRS=$(grep -oP '0x[0-9a-fA-F]+' "$LOGFILE" | sort | uniq)

for addr in $ADDRS; do
    # 去除前缀时输出更好看
    SYMBOL_INFO=$(addr2line -e "$BINARY" -f -C -p "$addr")
    echo "$addr -> $SYMBOL_INFO"
done