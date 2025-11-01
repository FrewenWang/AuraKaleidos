#!/bin/bash

search_file="data/misc/platform_aura2_cicd.txt"

if [ $# -lt 2 ] ; then
    echo "Usage: $0 <vendor_id> <hardware_name>"
    echo "e.g:"
    echo "bash init_devices.sh Qualcomm SM8550"
    exit 1
fi 

vendor_id=$1
hardware_name=$2

echo "Config: vendor_id: $vendor_id, hardware_name: $hardware_name, search_file: $search_file"

for sno in $(adb devices | grep ".device$" | awk '{print $1}')
do
    echo "Device: $sno"
    adb -s $sno wait-for-device root
    adb -s $sno wait-for-device remount

    adb -s $sno shell "rm $search_file"

    adb -s $sno shell "echo Vendor : $vendor_id > data/misc/platform_aura2_cicd.txt"
    adb -s $sno shell "echo Hardware : $hardware_name >> ${search_file}"
    adb -s $sno shell "echo Busy : Not >> ${search_file}"

    echo ""
    echo "${search_file}:"
    adb -s $sno shell "cat ${search_file}"

    echo "Device: $sno Done"

    echo ""
done

echo ""
echo "All Done"