# QNN 2.44 模型转换与逐层校验

本目录保存 QNN 2.44 的实验性转换流程，按步骤完成浮点运行、量化转换、设备推理和逐层
精度比较。脚本依赖 Qualcomm QNN SDK、Android NDK、ADB 目标设备、ONNX 模型和校准
RAW 数据，不适合在没有这些依赖的普通开发机上直接运行。

## 运行顺序

1. 复制 `jdd_env.env` 为本地环境文件，并通过环境变量设置模型名、项目目录、输入清单和量化参数
   改为当前实验值。不要提交包含个人绝对路径的本地配置。
2. 修改 `qnn_setup_244.env` 中的 SDK/NDK 根目录，或者在执行前从外部导出
   `QNN_SDK_ROOT`、`ANDROID_NDK_ROOT`。
3. 在本目录加载环境并依次运行：

```bash
cd FantasyHPC/aura-model-conveter/qnn244_convert
source qnn_setup_244.env
export ENV_ONNX_MODEL_NAME=<模型名称>
source jdd_env.env
bash step0_run_unquantized.sh
bash step1_qnn_converter.sh
python step2_qnn_batch_infer_quantized.py --help
python step3_check_qnn_perlayer.py
```

## 输入与输出

| 内容 | 约定 |
|---|---|
| ONNX 模型 | `${ENV_PROJECT_ROOT_DIR}/onnx/${ENV_ONNX_MODEL_NAME}.onnx` |
| 测试/校准清单 | `ENV_TEST_LIST`、`ENV_CALIB_LIST` |
| 量化覆盖配置 | `ENV_OVERRIDE_JSON_PATH` |
| 转换结果 | `${ENV_PROJECT_ROOT_DIR}/convert_output/` |
| 推理结果 | `${ENV_PROJECT_ROOT_DIR}/infer_output/` |

`step2` 会连接 Android/高通目标设备并推送模型，执行前必须先用 `adb devices` 确认设备。
当前 shell/env 文件仍保留历史机器路径，属于待参数化的技术债务；在完成代码改造前，应仅
通过未提交的本地副本修改这些值。
