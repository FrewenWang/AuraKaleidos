# Alice ONNX Runtime 推理示例

工程支持 macOS、Linux 和 Windows 本机构建，也可配合 Android/QNX 工具链交叉编译。平台变量遵循仓库的 `cmake/AuraPlatform.cmake`。

准备 OpenCV 4 和目标平台对应的 ONNX Runtime，然后执行：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
  -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
cmake --build build --parallel
```

仓库存在匹配 `<系统>-<架构>-release` 的预编译包时会自动选择，也可以通过 `AURA_OPENCV_ROOT` 和 `ONNXRUNTIME_ROOT` 覆盖。

运行：

```bash
./build/AliceBaiduFaceDetection model.onnx image.jpg
```

程序使用 CPU Execution Provider，读取单个静态 NCHW 图像输入，并打印全部输出 Tensor 的名称和形状。模型路径和图片路径均由命令行提供，不包含平台相关的绝对路径。
