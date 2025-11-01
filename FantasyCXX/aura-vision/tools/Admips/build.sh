#!/bin/bash
# 编译 Admips 工具
show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -r, --release           Set build type to Release [default]"
    echo "   -d, --debug             Set build type to Debug"
    echo "   -t, --target            Set build target, 1-android-armv7a, 2-android-armv8a, 3-android-x86_64, 4-android-x86"
    echo "   -h, --help              show help message"
    echo
    echo "example: ./build.sh -r -t 2"
}

# 输入参数解析
while [ $# != 0 ]
do
  case "$1" in
    -t)
        target_idx=$2
        shift
        ;;
    --target)
        target_idx=$2
        shift
        ;;
    -r)
        build_type="Release"
        ;;
    --release)
        build_type="Release"
        ;;
    -d)
        build_type="Debug"
        ;;
    --debug)
        build_type="Debug"
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

# 目标平台
android_target_type="Arm64_v8a"
arch_type="arm64-v8a"
case "$target_idx" in
1)
    android_target_type="Armeabi_v7a"
    arch_type="armeabi-v7a"
    ;;
2)
    android_target_type="Arm64_v8a"
    arch_type="arm64-v8a"
    ;;
3)
    android_target_type="X86_64"
    arch_type="x86_64"
    ;;
4)
    android_target_type="X86"
    arch_type="x86"
    ;;
*)
    android_target_type="Arm64_v8a"
    arch_type="arm64-v8a"
    ;;
esac

# 编译android工程
echo "===== 拷贝 libvision 到 jniLibs"
build_type_lower=$(echo ${build_type} | tr '[A-Z]' '[a-z]')
jniLibDir=app/src/main/jniLibs/${arch_type}
if [ ! -d ${jniLibDir} ]; then
  mkdir -p ${jniLibDir}
fi

cp ../../../build/android-${arch_type}-${build_type_lower}/install/lib/${arch_type}/*.so ${jniLibDir}
rm ${jniLibDir}/libc++_shared.so

echo "===== 开始编译 Admips"

./gradlew assemble${android_target_type}${build_type}

# 整理产出物
echo "===== 拷贝apk"
output_dir=output
rm -rf ${output_dir}
if [ ! -d ${output_dir} ]; then
    mkdir -p ${output_dir}
fi
cp app/build/outputs/apk/${android_target_type}/${build_type}/*.apk ${output_dir}/

echo "===== DONE!"