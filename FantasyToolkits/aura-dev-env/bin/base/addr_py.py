#!/usr/bin/env python3
import re
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
import logging

# 解析命令行参数
parser = argparse.ArgumentParser(description="解析 Android crash 堆栈，使用 addr2line 输出源码行号")
parser.add_argument("--log", required=True, help="堆栈日志文件路径")
parser.add_argument("--so-dir", required=True, help="包含 .so 文件的目录")
parser.add_argument("--arch", default="aarch64", help="架构（默认为 aarch64）")

args = parser.parse_args()
log_path = Path(args.log)
so_dir = Path(args.so_dir)

# 匹配堆栈行
stack_re = re.compile(r'#\d+\s+pc\s+([0-9a-f]+)\s+(/\S+\.so)')

# 记录待解析地址
addr_map = defaultdict(list)

# 读取堆栈日志并提取地址
with open(log_path, 'r') as f:
    for line in f:
        match = stack_re.search(line)
        if match:
            addr, so_path = match.groups()
            so_name = Path(so_path).name
            addr_map[so_name].append(addr)

# 执行 addr2line 查询每个地址
def resolve_addr(so_file, addr):
    try:
        result = subprocess.run(
            ['addr2line', '-f', '-C', '-e', str(so_file), addr],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8'
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR] {e}"

# 输出解析结果
print("📍 Address Resolution Result:")
for so_name, addr_list in addr_map.items():
    so_file = so_dir / so_name
    if not so_file.exists():
        print(f"\n❌ 找不到对应的 .so 文件: {so_file}")
        continue
    print(f"\n🔧 Resolving in {so_file}")
    for addr in addr_list:
        resolved = resolve_addr(so_file, f"0x{addr}")
        print(f"  【{addr}】 -> {resolved}")
