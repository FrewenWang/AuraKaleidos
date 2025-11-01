#!/bin/bash

res_dir="result"
if [ ! -d ${res_dir} ]; then
  mkdir ${res_dir}
fi

echo "pulling result from device..."
adb pull /sdcard/dmips_result.csv /sdcard/cpu_mem_result.csv ${res_dir}/