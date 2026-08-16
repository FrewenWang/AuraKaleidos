# AliceOpenCV

OpenCV Python Notebook/脚本与 C++ CMake demo。

```text
AliceOpenCV/
├── OpenCVPython/       # Python demo、Notebook、images/、utils/
└── OpenCVCXX/          # 多个独立 C++ demo target
```

Python 示例需 `opencv-python` 和 `numpy`，带 `imshow` 的脚本需桌面环境。C++ 构建：

```bash
cmake -S OpenCVCXX -B OpenCVCXX/build
cmake --build OpenCVCXX/build
```

CMake 使用 `find_package(OpenCV)`；如 OpenCV 不在标准位置，传入 `-DOpenCV_DIR=<opencv-config-dir>`。不再在工程中保留个人电脑路径。
