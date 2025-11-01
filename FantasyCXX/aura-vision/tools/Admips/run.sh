#!/bin/bash
# 安装并执行测试脚本
show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -m, --model             test mode(dmips or cpumem)"
    echo "   -i, --install           Install apk first"
    echo "   -h, --help              show help message"
    echo
    echo "example: ./run.sh -i -c dmips"
}

# 输入参数解析
install_apk=0
test_mode="dmips"
while [ $# != 0 ]
do
  case "$1" in
    -i)
        install_apk=1
        ;;
    --install)
        install_apk=1
        ;;
    -m)
        test_mode="$2"
        shift
        ;;
    --mode)
        test_mode="$2"
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

./set_device.sh 1

if [ $install_apk = 1 ];then
  adb install -r -t output/*.apk
fi

adb shell am force-stop com.baidu.admips

# setting device
echo "setting device..."
if [ $test_mode = "dmips" ]; then
  ./set_device.sh 2
elif [ $test_mode = "cpumem" ]; then
  ./set_device.sh 3
fi

adb shell am start -n com.baidu.admips/.MainActivity --es testMode ${test_mode}
echo "testing..."
