#!/bin/bash
# 设置车机
show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   1,  调试福特车机"
    echo "   2,  保留单个大核"
    echo "   3,  打开所有核"
    echo
    echo "example: ./set_device.sh 2"
}

# 输入参数解析
set_mode=1
while [ $# != 0 ]
do
  case "$1" in
    1)
        set_mode=1
        shift
        ;;
    2)
        set_mode=2
        shift
        ;;
    3)
        set_mode=3
        shift
        ;;
    -h)
        show_help
        exit 1
        ;;
    --help)
    show_help
        exit 1
        ;;
    *)
        ;;
  esac
  shift
done

case $set_mode in
    1)
		adb root
        adb remount
        adb shell setprop bdcarsec.pm.uninstall 0
        adb shell setprop bdcarsec.pm.install 0
        adb shell setprop bdcarsec.am.run.verifyprocess 0
        adb shell setprop bdcarsec.time.scan 0
        adb shell setenforce 0
        ;;
    2)
        # i.MX8 的大核
        adb root
        adb remount
        adb shell < close_cores.txt
        ;;
    3)
        adb root
        adb remount
        adb shell < open_cores.txt
        ;;
    *)
        ;;
esac