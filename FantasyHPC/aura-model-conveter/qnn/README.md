


# QNN模型转换工具使用指南


## 环境准备
设置ONNX和QNN环境变量：
```shell
# 环境变量设置
export ONNX_ROOT=/home/baiduiov/work/inference/onnx/onnx-1.10.2/onnx # 1.10.2 # 1.6.0
export QNN_SDK_ROOT=/home/baiduiov/work/inference/qnn/qnn-v2.5.0.221123101258_42157-auto
export QNN_SDK_BIN=${QNN_SDK_ROOT}/target/x86_64-linux-clang/bin
```

# 模型准备

按照自己模型存放的位置修改如下参数

```shell
# 设置模型转换的类型：int8 、float16
enable_convert_type="float16"
# 设置ONNX原始模型的存放根目录
model_path="/home/baiduiov/work/vision-space/OriginalModel/FaceCall/20230807v15_qat/onnx"
# 设置原始ONNX模型的名称，不要后缀
model_name="DMSCall230809V14MainQAT"
# 在算法用于对齐的RAW数据中随便选一张，和原始模型放在同一路径下面
test_raw="call.raw"
# 设置参与量化的500张RAW数据的根目录
input_raw_list="/home/baiduiov/work/vision-space/OriginalModel/FaceCall/20230807v15_qat/500/raw"
input_height=48
input_width=64
input_channel=1
just_compare="n" # y or n
```


# 开始转换

修改完main.sh之后的相关模型配置，我们直接执行main.sh的脚本


# 生成数据

```shell
.
├── OnnxResult
│         ├── batch_norm_20.tmp_2.raw
├── QnnModel
│         ├── input_list_quant.txt
│         ├── input_list_run.txt
│         └── int8
│         ├── cpp
│       │   ├── DMSCall230714V13MainQAT.bin
│       │   ├── DMSCall230714V13MainQAT.cpp
│       │   └── DMSCall230714V13MainQAT_net.json
│       └── so
│           └── x86_64-linux-clang
│               ├── DMSCall230714V13MainQAT_int8.bin
│               └── libDMSCall230714V13MainQAT.so
└── QnnResult
    └── Result_0
        ├── batch_norm_22_tmp_2.raw
        ├── batch_norm_24_tmp_2.raw
  

```

生成的数据解释：

生成的QNN模型路径：
/QnnModel/cpp/so/x86_64-linux-clang/DMSCall230714V13MainQAT_int8.bin


