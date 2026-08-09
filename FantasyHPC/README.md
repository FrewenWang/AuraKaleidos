# FantasyHPC

高性能计算、模型优化与部署工程集合。

- `alice-model-quant`：量化与 ONNX 图处理，脚本、模型、数据、日志分别归档。
- `aura-model-converter`：按 ONNX、TensorRT、QNN 后端组织的转换工具。
- `aura-tensorrt`：TensorRT 示例，只在具备 NVIDIA 环境的平台运行。
- `model-distillation`：`src/` 模型实现与 `scripts/` 训练入口。
- `model_deploy`：模型服务部署示例。
- `alice-jetson-inference`：上游 Jetson 工程镜像，保留其原生目录。

模型和大型数据不使用 Git LFS，统一由 `.gitignore` 排除并在各项目 README 中说明获取方式。
