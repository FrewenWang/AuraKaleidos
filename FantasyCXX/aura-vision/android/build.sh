#!/bin/bash

# =============================================================
# Android工程路径（当前路径）
vision_android_dir=$(pwd)

# 默认的native工程路径，可根据实际修改
vision_native_dir=$vision_android_dir/../..

echo "VISION_NATIVE_DIR=$vision_native_dir"
# 默认编译armv8a
# 1-android-armv7a
# 2-android-armv8a
# 3-android-x86_64
# 4-android-x86
# 5-android-armv7a and x86 (For QA)
# 6-android-armv8a and x86 (For QA)

# 默认Release模式
build_type="Release"

# 默认使用内部模型数据（不使用外部模型文件）
use_external_model=false

# 默认不使用 Baidu_protect 的加密
use_external_encrypt=false
# =============================================================


show_help() {
    echo "Usage: $0 [option...]" >&2
    echo
    echo "   -r, --release           Set build type to Release [default]"
    echo "   -d, --debug             Set build type to Debug"
    echo "   -b, --build_type        Set build type [debug,release]"
    echo "   --RelWithDebInfo        Set build type to RelWithDebInfo"
    echo "   -t, --target            Set build target, 1-android-armv7a, 2-android-armv8a, 3-android-x86_64, 4-android-x86"
    echo "   --ext_model             use external model file"
    echo "   --ext_encrypt           use external encryption"
    echo "   -p, --product           Set build product, for example: ford_cd542, toyota_760, toyota_030d,
                                     chery, hyundai_SGEN5WxC, evergrande, honda_23m ..."
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
    -p)
        PRODUCT=$2
        shift
        ;;
    --product)
        PRODUCT=$2
        shift
        ;;
    -b)
        build_type=$2
        shift
        ;;
    --build_type)
        build_type=$2
        shift
        ;;
    -r)
        build_type="release"
        ;;
    --release)
        build_type="release"
        ;;
    -d)
        build_type="debug"
        ;;
    --debug)
        build_type="debug"
        ;;
    --RelWithDebInfo)
        build_type="relWithDebInfo"
        ;;
    --ext_model)
        use_external_model=true
        ;;
    --ext_encrypt)
        use_external_encrypt=true
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
android_target_type=Arm64_v8a
native_target_type=android-arm64-v8a

case "$target_idx" in
1)
    native_target_type="android-armeabi-v7a"
    android_target_type="Armeabi_v7a"
    ;;
2)
    native_target_type="android-arm64-v8a"
    android_target_type="Arm64_v8a"
    ;;
3)
    native_target_type="android-x86_64"
    android_target_type="X86_64"
    ;;
4)
    native_target_type="android-x86"
    android_target_type="X86"
    ;;
5)
    native_target_type="android-armeabi-v7a_x86"
    android_target_type="Armv7a_x86"
    ;;
6)
    native_target_type="android-armeabi-v8a_x86"
    android_target_type="Armv8a_x86"
    ;;
*)
    native_target_type="android-arm64-v8a"
    android_target_type="Arm64_v8a"
    ;;
esac

android_target_type_lower=$(echo $android_target_type | tr '[A-Z]' '[a-z]')
build_type_lower=$(echo $build_type | tr '[A-Z]' '[a-z]')

# 判断JNILIBS路径是否存在，不存在则创建
android_jnilibs=vision_ability/src/main/jniLibs
android_jnilibs_armeabi_v7a=${android_jnilibs}/armeabi-v7a
android_jnilibs_arm64_v8a=${android_jnilibs}/arm64-v8a
android_jnilibs_x86_64=${android_jnilibs}/x86_64
android_jnilibs_x86=${android_jnilibs}/x86
for dir in $android_jnilibs_armeabi_v7a $android_jnilibs_arm64_v8a $android_jnilibs_x86_64 $android_jnilibs_x86 ; do
    if [ ! -d $dir ]; then
      mkdir $dir
    else
      rm -r $dir/*
    fi
done

echo "===== 拷贝so..."
if [ $target_idx -eq 5 ]; then
    visionLibPath_ArmV7=$vision_native_dir/build/android-armeabi-v7a-release/install/lib
    visionLibPath_X86=$vision_native_dir/build/android-x86-release/install/lib
    cp -rf ${visionLibPath_ArmV7}/*.so vision_ability/src/main/jniLibs/armeabi-v7a/
    cp -rf ${visionLibPath_X86}/*.so vision_ability/src/main/jniLibs/x86/

elif [ $target_idx -eq 6 ]; then
    visionLibPath_Arm64=$vision_native_dir/build/android-arm64-v8a-release/install/lib
    visionLibPath_X86=$vision_native_dir/build/android-x86-release/install/lib
    cp -rf ${visionLibPath_Arm64}/*.so vision_ability/src/main/jniLibs/arm64-v8a/
    cp -rf ${visionLibPath_X86}/*.so vision_ability/src/main/jniLibs/x86/

else
    if [ $android_target_type_lower = arm64_v8a ]; then
        jnilib_path="arm64-v8a"
    elif [ $android_target_type_lower = armeabi_v7a ]; then
        jnilib_path="armeabi-v7a"
    else
        jnilib_path=${android_target_type_lower}
    fi

    visionLibPath=${vision_native_dir}/build/${native_target_type}-${build_type}/install/lib
    echo "visionLibPath == $visionLibPath";
    if [ -d $visionLibPath ]; then
      cp -rf ${visionLibPath}/*.so vision_ability/src/main/jniLibs/${jnilib_path}/
    else
      echo "===== Native libs NOT FOUND! EXIT!";
      exit
    fi

    if [ "$use_external_model" = true ]; then
      echo "===== 拷贝model文件..."
      cp $vision_native_dir/models/generated/vision_model* vision_ability/src/main/assets/model/
    else
      echo "===== 删除model文件..."
      rm -rf vision_ability/src/main/assets/model/vision_model.bin
    fi
fi

# 3，编译android工程
echo "===== 开始编译vision-ability-android..."
echo "===== 目标平台: ${android_target_type_lower}"
echo "===== 编译类型: ${build_type}"
./gradlew assemble${android_target_type}${build_type}
echo "===== 结束编译vision-ability-android"

# 4，整理产出物
echo "===== 拷贝交付物到output..."
time_stamp=`date "+%Y%m%d%H%M%S"`
git_branch_name=`git symbolic-ref --short -q HEAD`
# 使用当前分支名+时间戳 作为版本名称
git_branch_name=${git_branch_name##dev_}
artifact_path=vision_ability-${git_branch_name}-${PRODUCT}-${android_target_type_lower}_${time_stamp}
#output_sdk_path=${vision_native_dir}/build/${native_target_type}-${build_type}/output_sdk/${artifact_path}
output_sdk_path=${vision_native_dir}/build/${native_target_type}-${build_type}/output_sdk
rm -rf ${output_sdk_path}
if [ ! -d ${output_sdk_path} ]; then
    mkdir -p ${output_sdk_path}
fi

# aar
output_aar_dir=${output_sdk_path} #/aar
if [ ! -d ${output_aar_dir} ]; then
    mkdir -p ${output_aar_dir}
fi
built_aar_path=vision_ability/build/outputs/aar
build_aar_name=vision_ability-${android_target_type_lower}-${build_type_lower}
output_aar_name=vision_ability_${PRODUCT}_${git_branch_name}_${android_target_type_lower}_${build_type_lower}_${time_stamp}
#cp ${built_aar_path}/${build_aar_name}.aar ${output_aar_dir}/${output_aar_name}-${time_stamp}.aar
cp ${built_aar_path}/${build_aar_name}.aar ${output_aar_dir}/${output_aar_name}.aar

# jar
output_jar_dir=${output_sdk_path} #/jar
if [ ! -d ${output_jar_dir} ]; then
    mkdir -p ${output_jar_dir}
fi
built_jar_path=vision_ability/build/intermediates/aar_main_jar/${android_target_type_lower}${build_type}/classes.jar
#built_so_path=vision_ability/build/intermediates/stripped_native_libs/${android_target_type_lower}${build_type}/out/lib/${jnilib_path}/*.so
built_so_path=vision_ability/build/intermediates/stripped_native_libs/${android_target_type_lower}${build_type}/out/lib
#jar_name=vision_ability-${android_target_type_lower}-${build_type_lower}-${time_stamp}.jar
jar_name=VisionAbility_${PRODUCT}_${git_branch_name}_${android_target_type_lower}_${build_type_lower}_${time_stamp}.jar
cp ${built_jar_path} ${output_jar_dir}/${jar_name}
#cp ${built_so_path} ${output_jar_dir}/
cp ${built_so_path}/"arm64-v8a"/*.so ${output_jar_dir}/
cp ${built_so_path}/"x86"/*.so ${output_jar_dir}/