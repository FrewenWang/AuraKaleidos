# macOS 工程运行与测试

本仓库包含桌面、移动端、嵌入式、GPU 和模型训练等多种工程。统一测试入口只覆盖能够在当前 macOS 主机上原生构建、且不依赖专用硬件、私有 SDK、大模型或外部数据集的工程。

跨平台 CMake 变量、工具链用法和依赖目录约定见 `docs/CROSS_PLATFORM_BUILD.md`。

## 快速开始

```bash
./scripts/bootstrap_macos.sh
./scripts/test_macos.sh
```

`bootstrap_macos.sh` 创建根目录 `.venv`，安装 Python 测试依赖，并安装各 Node.js 工程依赖。`test_macos.sh` 在 `build/macos-tests` 中构建 C/C++ 工程，然后执行 CTest、Python unittest、Node.js 测试和 Shell 语法检查。以上生成目录都已被 Git 忽略。

## 已纳入自动测试

| 工程 | macOS 验证内容 |
| --- | --- |
| `FantasyCXX` | CMake 构建和 3 个 CTest 测试；设备视觉模块默认关闭 |
| `FantasyAlgorithm/CXX` | 股票利润算法库及边界条件单测 |
| `FantasyAI/AliceOpenCV/OpenCVCXX` | OpenCV 图像工具单测及命令行端到端测试 |
| `FantasyAlgorithm/python` | 删除元素、排序数组去重单测 |
| `FantasyPython/alice-auto-driving` | 马氏距离关联和 RANSAC 单测 |
| `FantasyPython/aura-data-compare` | 数组误差比较单测 |
| `FantasyPython/aura-pyutils` | 文件后缀工具单测 |
| `FantasyAutoDrive/kalman_filter_with_yolo11_objects_tracker` | 不加载 YOLO 权重的 Kalman Filter 单测 |
| `FantasyJS` | 浏览器键码工具测试 |
| `FantasyNodeJS` 中 8 个 Node 工程 | 模块、路由、HTTP 和 HTML 解析测试；爬虫测试不访问公网 |
| `FantasyNodeJS/NyxTSExpress` | Sass/TypeScript 构建、lint 和 Jest 单测 |
| `FantasyShell` | Bash 语法检查 |

## 暂不纳入 macOS 自动测试

- `FantasyAndroid`、Android 示例：需要 Android SDK、模拟器或设备。
- `FantasyCuda`：需要 NVIDIA CUDA 工具链和 GPU，macOS 不支持当前 CUDA 运行时。
- `FantasyHPC/alice-jetson-inference`：面向 NVIDIA Jetson。
- `FantasyCXX/aura-vision`、`aura-vision-hpc`：依赖 Android、NCNN、SNPE、QNN 或设备 OpenCL/NEON 环境。
- iOS、Flutter 工程：目标运行环境是 iOS/移动端，未加入本机命令行测试矩阵。
- AI 训练、Notebook、模型推理示例：依赖大型数据集、模型权重、PyTorch/ONNX Runtime 或长时间训练，保留为手动验证项目。
- 网络爬虫的真实站点访问、MongoDB 集成：自动测试使用本地固定输入，避免网络波动和外部服务影响结果。

这些跳过项不是判定为代码错误，而是当前 macOS 通用环境不具备对应运行条件。获得目标平台工具链或测试资源后，应为它们建立独立 CI 测试矩阵。
