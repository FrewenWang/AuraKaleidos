# Alice OpenCV C++

跨平台 OpenCV C++ 示例工程。

- `src/`：图像工具库和主 CLI。
- `tests/`：无 GUI 单元测试与 CLI 测试。
- `examples/`：独立教学程序，全部使用输入/输出参数，不含本机绝对路径。
- `assets/images/`：小型示例图片。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_OPENCV_EXAMPLES=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/OpenCVDemo assets/images/image1.jpg build/result.jpg
```

OpenCV 依赖选择遵循仓库根目录 `cmake/AuraOpenCV.cmake`。
