#!/usr/bin/env bash

# 使用以下开关配置某个模型转换是否开启
# enable_xxx=[0|1]， 0 = disable, 1 = enable

# 环境变量设置
# 下载地址：https://github.com/onnx/onnx/releases?page=2
export ONNX_ROOT=/home/frewen/Library/onnx-1.10.2/onnx # 1.10.2 # 1.6.0
export QNN_SDK_ROOT=/home/frewen/Library/qnn-v2.5.0.221123101258_42157-auto
export QNN_SDK_BIN=${QNN_SDK_ROOT}/target/x86_64-linux-clang/bin

# 设置模型转换的类型：int8 、float16
enable_convert_type="float16"
# 设置ONNX原始模型的存放根目录
model_path="/home/frewen/03.ProgramSpace/20.AI/04.Resource/OriginModels/20240903_单通道_首层融合_非量化/onnx/"
# 设置原始ONNX模型的名称，不要后缀
model_name="FaceDetection20240903V7Main"
# 在算法用于对齐的RAW数据中随便选一张，和原始模型放在同一路径下面
test_raw="input_224x320.raw"
# 设置参与量化的500张RAW数据的根目录
input_raw_list="/home/baiduiov/work/vision-space/OriginalModel/PlayPhone/20230908V15_量化_0912/600张图片和二进制文件/playphone_erjinzhi/erjinzhi"
input_height=224
input_width=320
input_channel=1
just_compare="n" # y or n

# 转换 量化模型 - Int8  版本
if [[ $enable_convert_type == "int8" ]]; then
  echo "[ start convert_model_int8 ]"
  ./converter.sh \
    -mt 8 \
    -md $model_path \
    -mn $model_name \
    -tr $test_raw \
    -trd $input_raw_list \
    -ih $input_height \
    -iw $input_width \
    -ic $input_channel \
    -onnx y \
    -qnn y \
    -jcp $just_compare
fi

# 转换 Gesture Landmark - Float 版本
if [[ $enable_convert_type == "float16" ]]; then
  echo "[ start convert_model_f16 ]"
  ./converter.sh \
    -mt 16 \
    -md $model_path \
    -mn $model_name \
    -tr $test_raw \
    -trd $input_raw_list \
    -ih $input_height \
    -iw $input_width \
    -ic $input_channel \
    -onnx y \
    -qnn y \
    -jcp $just_compare
fi
