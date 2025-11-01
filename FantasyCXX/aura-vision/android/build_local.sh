#!/bin/bash

# vision本地打包工具说明

# 如果编译armv7a 应将参数改为 -t 1；相应地，x86_64: -t 3; armv8a: -t 2
# 如果不需要编译native工程，则去掉 -n 选项
# 如果不需要从native拷贝so和模型，则去掉 -c 选项
# 如果编译debug模式，改为选项 -d
# 如果需要自动切出native某个分支进行编译,可以加上-g -b branchname (例如 -g -b dev_v0.7.4)，可能造成冲突，慎用

# 编译完成的aar和apk会自动拷贝到output目录下

# 以下命令首先编译native armv8a，然后拷贝so和模型，然后以release模式编译android armv8a
#./build.sh -r -t 2 -c -n

# 以下命令编译native x86_64， 需要编译x86_64可打开
# ./build.sh -r -t 3 -c -n

# 以下命令编译native armeabi-v7a， 需要编译armv7a可打开
# ./build.sh -r -t 1 -c -n

echo -e "
你想要执行的操作?
==============================================
1.android_armeabi_v7a
2.android_arm64-v8a
3.android_x86_64
4.android_x86
5.android_armeabi_v7a && android_x86
=============================================="
printf "Input integer number: "
read num
type=2
case $num in
    1)
        type=1
    ;;
    2)
        type=2
    ;;
    3)
        type=3
    ;;
    4)
        type=4
    ;;
    5)
        type=5
    ;;
    *)
        type=2
    ;;
esac
echo "======"${type}
#./build.sh -r -t ${type} -c -n
./build.sh -d -t ${type} -c -n