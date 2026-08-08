# 跨平台 CMake 构建

仓库内的 C++ 工程共享 `cmake/AuraPlatform.cmake`。平台选择方式与 `FantasyCXX` 保持兼容，同时优先使用 CMake 工具链提供的目标系统信息。

## 平台变量

| 变量 | 说明 | 默认来源 |
| --- | --- | --- |
| `HOST_OS` | 编译主机系统 | `CMAKE_HOST_SYSTEM_NAME` |
| `HOST_ARCH` | 编译主机架构 | `CMAKE_HOST_SYSTEM_PROCESSOR` |
| `TARGET_OS` | 目标系统：`mac`、`linux`、`windows`、`android`、`ios`、`qnx` | `CMAKE_SYSTEM_NAME` |
| `TARGET_ARCH` | 目标架构或 Android ABI | `ANDROID_ABI` 或 `CMAKE_SYSTEM_PROCESSOR` |
| `AURA_DEPENDENCY_VARIANT` | 预编译依赖类型：`release` 或 `debug` | `release` |
| `AURA_ENABLE_WARNINGS` | 启用 MSVC/Clang/GCC 常用警告 | `ON` |
| `AURA_WARNINGS_AS_ERRORS` | 将警告视为错误 | `OFF` |

系统和架构名称会被统一，例如 `Darwin`、`macOS`、`osx` 转为 `mac`，`AMD64` 转为 `x86_64`，`aarch64` 转为 `arm64`。显式传入的 Android ABI（如 `arm64-v8a`）会保持不变。

平台模块同时保留 `FantasyCXX` 已使用的变量：`AURA_BUILD_MAC`、`AURA_BUILD_LINUX`、`AURA_BUILD_WINDOWS`、`AURA_BUILD_ANDROID`、`AURA_BUILD_IOS`、`AURA_BUILD_QNX`，以及对应的编译宏。

## 本机构建

通常不需要手工指定系统和架构：

```bash
cmake -S FantasyAlgorithm/CXX -B build/algorithm -DCMAKE_BUILD_TYPE=Release
cmake --build build/algorithm --parallel
ctest --test-dir build/algorithm --output-on-failure
```

## 使用工具链交叉编译

交叉编译时应让工具链设置 `CMAKE_SYSTEM_NAME` 和 `CMAKE_SYSTEM_PROCESSOR`。只有兼容旧脚本或选择特定 ABI 时才额外设置 `TARGET_OS/TARGET_ARCH`。

Android 示例：

```bash
cmake -S FantasyAI/AliceOpenCV/OpenCVCXX \
  -B build/opencv-android-arm64 \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DAURA_DEPENDENCY_VARIANT=release
cmake --build build/opencv-android-arm64 --parallel
```

Windows 和 Linux 使用对应编译器或工具链：

```text
cmake -S <project> -B <build-dir> -DCMAKE_TOOLCHAIN_FILE=<toolchain.cmake>
cmake --build <build-dir> --config Release --parallel
```

不要仅通过 `-DTARGET_OS=windows` 把 macOS 编译器当作 Windows 编译器；`TARGET_OS` 负责工程条件和依赖选择，真正的交叉编译能力由工具链提供。

## 第三方依赖

预编译包统一使用：

```text
<version>/<TARGET_OS>-<TARGET_ARCH>-<AURA_DEPENDENCY_VARIANT>
```

例如：

```text
v4.11.0/mac-x86_64-release
v4.11.0/android-arm64-v8a-release
```

OpenCV 示例优先选择匹配的仓库内预编译包。Linux、Windows 或其他未内置的平台，可传入：

```text
-DAURA_OPENCV_ROOT=<OpenCV 安装根目录>
```

也可以直接传入标准的 `OpenCV_DIR`。ONNX Runtime 示例不再包含 Linux 绝对路径，通过环境变量或 CMake 参数设置：

```text
ONNXRUNTIME_ROOT=<ONNX Runtime 安装根目录>
```

当依赖包与目标系统或架构不匹配时，配置阶段会给出明确错误，不会回退到其他架构的二进制。

## 测试策略

本机构建会创建并运行 CTest。`CMAKE_CROSSCOMPILING` 为真时仍会编译测试目标，但不会在宿主机注册或运行目标平台可执行文件；它们应在模拟器、设备或目标系统 CI 中运行。
