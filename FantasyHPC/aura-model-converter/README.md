# aura-model-converter

模型格式转换工具，按 `onnx/`、`tensorrt/` 和 `qnn/` 后端分组。脚本应从自身目录解析模板和输入文件；
转换后的模型统一写入 `models/` 或 `outputs/`，并由 Git 忽略。
