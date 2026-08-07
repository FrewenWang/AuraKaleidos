# FantasyCXX 运行设置指南

## 目录

1. [环境准备](#环境准备)
2. [项目构建](#项目构建)
3. [模块配置](#模块配置)
4. [运行示例](#运行示例)
5. [测试验证](#测试验证)
6. [开发环境设置](#开发环境设置)
7. [常见问题](#常见问题)

## 环境准备

### 1. 系统要求检查

#### 硬件要求
- **CPU**：x86_64 或 ARM64 架构
- **内存**：至少 4GB RAM（推荐 8GB 以上）
- **磁盘空间**：至少 2GB 可用空间
- **GPU**（可选）：支持 OpenCL 的显卡

#### 软件要求

**基础工具**：
```bash
# 检查 CMake 版本
cmake --version
# 要求：3.10 或更高版本

# 检查编译器
g++ --version
# 或
clang++ --version
# 要求：支持 C++17
```

### 2. 依赖安装

#### Ubuntu/Debian

```bash
# 更新包管理器
sudo apt update

# 安装基础编译工具
sudo apt install -y build-essential cmake git

# 安装可选依赖
sudo apt install -y libopencv-dev  # OpenCV
sudo apt install -y libgtest-dev   # Google Test
sudo apt install -y libeigen3-dev  # Eigen
```

#### macOS

```bash
# 使用 Homebrew
brew install cmake git

# 安装可选依赖
brew install opencv
brew install eigen
```

#### Windows

**Visual Studio**：
1. 下载并安装 [Visual Studio 2019](https://visualstudio.microsoft.com/vs/)
2. 安装 "使用 C++ 的桌面开发" 工作负载
3. 安装 [CMake](https://cmake.org/download/)

**依赖库**：
- 下载预编译的 OpenCV 库
- 或从源码编译所需依赖

### 3. 第三方库准备

#### OpenCV 安装

**Ubuntu**：
```bash
sudo apt install -y libopencv-dev
```

**macOS**：
```bash
brew install opencv
```

**源码编译**：
```bash
# 下载 OpenCV
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip

# 编译安装
cd opencv-4.x
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

#### Android NDK（移动端开发）

```bash
# 下载 Android NDK
# 设置环境变量
export ANDROID_NDK_ROOT=/path/to/android-ndk
export PATH=$ANDROID_NDK_ROOT:$PATH
```

## 项目构建

### 1. 获取项目代码

```bash
# 克隆项目
git clone <repository_url>
cd FantasyCXX

# 检查项目结构
ls -la
```

### 2. 使用构建脚本（推荐）

#### Linux/macOS 构建

```bash
# 赋予执行权限
chmod +x build.sh

# 构建当前平台（调试版本）
./build.sh -d

# 构建当前平台（发布版本）
./build.sh -r

# 查看帮助
./build.sh -h
```

#### Android 构建

```bash
# 构建 arm64-v8a 调试版本
./build.sh -t 2 -d

# 构建 arm64-v8a 发布版本
./build.sh -t 2 -r

# 构建 armeabi-v7a 调试版本
./build.sh -t 1 -d
```

### 3. 手动 CMake 构建

#### 创建构建目录

```bash
# 创建并进入构建目录
mkdir build && cd build

# 配置项目（Linux/macOS）
cmake ..

# 配置项目（Windows）
cmake -G "Visual Studio 16 2019" -A x64 ..
```

#### 常用构建选项

```bash
# 调试版本
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 发布版本
cmake -DCMAKE_BUILD_TYPE=Release ..

# 构建静态库
cmake -DBUILD_STATIC_LIB=ON ..

# 禁用示例构建
cmake -DBUILD_EXAMPLES=OFF ..

# 启用单元测试
cmake -DBUILD_UNIT_TEST=ON ..

# 指定 OpenCV 路径
cmake -DOpenCV_DIR=/path/to/opencv/build ..
```

#### 编译项目

```bash
# Linux/macOS
make -j$(nproc)

# Windows
cmake --build . --config Release

# 安装
make install
```

### 4. 构建验证

```bash
# 检查构建结果
ls build/

# 验证可执行文件
./build/examples/01.demo_sizeof

# 检查库文件
ls build/lib/
```

## 模块配置

### 1. aura-cv 配置

#### 基础配置

```cpp
// 初始化配置
AuraCV::Config config;
config.enable_opencl = true;        // 启用 OpenCL 加速
config.enable_neon = true;          // 启用 ARM NEON
config.enable_hexagon = false;      // Hexagon DSP
config.log_level = "INFO";          // 日志级别

// 初始化
AuraCV::Init(config);
```

#### 环境变量配置

```bash
# OpenCL 配置
export AURA_OPENCL_DEVICE=GPU
export AURA_OPENCL_PLATFORM=0

# 日志配置
export AURA_LOG_LEVEL=DEBUG
export AURA_LOG_FILE=aura_cv.log
```

### 2. aura-config-parser 配置

#### 配置文件示例

**config.ini**：
```ini
[application]
name = FantasyCXX
version = 1.0.0

[network]
host = localhost
port = 8080
timeout = 30

[logging]
level = INFO
file = app.log
max_size = 10485760

[vision]
model_path = models/
detection_threshold = 0.5
max_detections = 100
```

#### 程序内配置

```cpp
AuraConfig::Parser parser;

// 设置默认值
parser.SetDefault("network", "port", "8080");
parser.SetDefault("vision", "threshold", "0.5");

// 加载配置
if (!parser.ParseFile("config.ini")) {
    std::cerr << "Failed to load config" << std::endl;
    return -1;
}
```

### 3. 性能优化配置

#### 内存配置

```bash
# 内存池大小
export AURA_MEMORY_POOL_SIZE=1073741824  # 1GB

# 缓存大小
export AURA_IMAGE_CACHE_SIZE=536870912   # 512MB
```

#### 线程配置

```bash
# 线程池大小
export AURA_THREAD_POOL_SIZE=8

# OpenMP 线程数
export OMP_NUM_THREADS=4
```

## 运行示例

### 1. 基础示例运行

#### 数据类型演示

```bash
# 编译示例
cd build
make 01.demo_sizeof

# 运行
./examples/01.demo_sizeof

# 预期输出
# ====================普通数据结构sizeof字节数===============================
# sizeof(char)==1
# sizeof(short)==2
# sizeof(int)==4
# ...
```

#### 配置解析示例

```bash
# 创建配置文件
echo -e "[test]\nkey1=value1\nkey2=123" > test.ini

# 运行示例
./examples/ini_parser_demo test.ini
```

### 2. 视觉处理示例

#### 图像预处理

```bash
# 准备测试图像
cp /path/to/test.jpg ./test_image.jpg

# 运行图像处理示例
./examples/image_processing test_image.jpg
```

#### 目标检测

```bash
# 下载预训练模型
wget -O models/ssd_mobilenet.caffe https://example.com/models/ssd_mobilenet.caffe

# 运行检测示例
./examples/object_detection test_image.jpg models/ssd_mobilenet.caffe
```

### 3. 性能测试示例

```bash
# 运行性能基准测试
./examples/performance_benchmark

# 输出示例
# [PERF] Image resize: 1.23ms
# [PERF] Face detection: 15.67ms
# [PERF] Total pipeline: 18.45ms
```

## 测试验证

### 1. 单元测试

#### 构建测试

```bash
# 启用测试构建
cmake -DBUILD_UNIT_TEST=ON ..
make
```

#### 运行测试

```bash
# 运行所有测试
cd test
./run_tests

# 运行特定测试
./run_tests --gtest_filter="*Config*"

# 生成测试报告
./run_tests --gtest_output="json:test_report.json"
```

### 2. 集成测试

#### 测试脚本

```bash
#!/bin/bash
# test_integration.sh

echo "Running integration tests..."

# 测试配置解析
./test/config_parser_test
if [ $? -ne 0 ]; then
    echo "Config parser test failed"
    exit 1
fi

# 测试图像处理
./test/image_processing_test
if [ $? -ne 0 ]; then
    echo "Image processing test failed"
    exit 1
fi

echo "All integration tests passed!"
```

#### 自动化测试

```bash
# 运行完整测试套件
./scripts/run_all_tests.sh

# 生成覆盖率报告
./scripts/generate_coverage.sh
```

### 3. 性能测试

#### 基准测试

```bash
# 运行性能基准
./benchmarks/performance_benchmark

# 内存使用测试
valgrind --tool=massif ./examples/memory_test
ms_print massif.out.* > memory_report.txt
```

#### 压力测试

```bash
# 长时间运行测试
./scripts/stress_test.sh 3600  # 运行 1 小时

# 并发测试
./scripts/concurrent_test.sh 10  # 10 个并发线程
```

## 开发环境设置

### 1. IDE 配置

#### Visual Studio Code

**c_cpp_properties.json**：
```json
{
    "configurations": [
        {
            "name": "FantasyCXX",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/aura-cv/include",
                "${workspaceFolder}/aura-utils/include"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "gcc-x64"
        }
    ],
    "version": 4
}
```

**tasks.json**：
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "./build.sh",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        }
    ]
}
```

#### CLion 配置

1. 打开项目根目录
2. 配置 CMake 选项：
   - CMake options: `-DBUILD_EXAMPLES=ON -DBUILD_UNIT_TEST=ON`
   - Build directory: `build`
3. 配置运行/调试配置

### 2. 调试设置

#### GDB 调试

```bash
# 编译调试版本
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# 启动调试
gdb ./examples/demo_program

# GDB 命令
(gdb) break main
(gdb) run
(gdb) next
(gdb) print variable_name
```

#### Valgrind 内存检查

```bash
# 内存泄漏检查
valgrind --leak-check=full ./examples/demo_program

# 性能分析
valgrind --tool=callgrind ./examples/demo_program
callgrind_annotate callgrind.out.*
```

### 3. 代码格式化

#### Clang-Format

```bash
# 检查代码格式
clang-format --style=file -n src/*.cpp

# 自动格式化
clang-format --style=file -i src/*.cpp include/*.h

# 批量格式化
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

#### 格式化脚本

```bash
#!/bin/bash
# format_code.sh

echo "Formatting code..."

# 格式化所有 C++ 文件
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
    -exec clang-format -i {} \;

echo "Code formatting completed!"
```

## 常见问题

### 构建问题

#### Q1: CMake 找不到 C++ 编译器

**错误信息**：
```
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

**解决方案**：
```bash
# 检查编译器安装
which g++
which clang++

# 重新安装编译器
sudo apt install build-essential  # Ubuntu
brew install gcc                   # macOS
```

#### Q2: 找不到 OpenCV

**错误信息**：
```
Could NOT find OpenCV (missing: OpenCV_LIBS)
```

**解决方案**：
```bash
# 安装 OpenCV
sudo apt install libopencv-dev     # Ubuntu
brew install opencv                # macOS

# 或指定 OpenCV 路径
cmake -DOpenCV_DIR=/path/to/opencv/build ..

# 或禁用 OpenCV
cmake -DBUILD_OPENCV=OFF ..
```

#### Q3: 内存不足

**错误信息**：
```
virtual memory exhausted: Cannot allocate memory
```

**解决方案**：
```bash
# 减少并行编译数
make -j2  # 而不是 make -j$(nproc)

# 增加交换空间
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 运行时问题

#### Q1: 动态库加载失败

**错误信息**：
```
error while loading shared libraries: libaura_cv.so: cannot open shared object file
```

**解决方案**：
```bash
# 设置库路径
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH

# 或重新链接
sudo ldconfig

# 或构建静态库
cmake -DBUILD_STATIC_LIB=ON ..
```

#### Q2: OpenCL 设备未找到

**错误信息**：
```
OpenCL device not found
```

**解决方案**：
```bash
# 检查 OpenCL 安装
clinfo

# 安装 OpenCL ICD
sudo apt install ocl-icd-opencl-dev  # Ubuntu

# 或禁用 OpenCL
export AURA_ENABLE_OPENCL=0
```

#### Q3: 模型文件未找到

**错误信息**：
```
Model file not found: models/face_detection.caffe
```

**解决方案**：
```bash
# 下载模型文件
mkdir -p models
wget -O models/face_detection.caffe https://example.com/models/face_detection.caffe

# 或指定模型路径
export AURA_MODEL_PATH=/path/to/models
```

### 性能问题

#### Q1: 处理速度慢

**可能原因**：
- 未启用硬件加速
- 图像分辨率过高
- 内存带宽限制

**优化方案**：
```bash
# 启用硬件加速
export AURA_ENABLE_OPENCL=1
export AURA_ENABLE_NEON=1

# 降低处理分辨率
export AURA_MAX_IMAGE_WIDTH=640
export AURA_MAX_IMAGE_HEIGHT=480

# 使用性能分析工具
valgrind --tool=callgrind ./your_program
```

#### Q2: 内存使用过高

**优化方案**：
```bash
# 限制内存池大小
export AURA_MEMORY_POOL_SIZE=536870912  # 512MB

# 启用内存监控
export AURA_ENABLE_MEMORY_MONITOR=1

# 使用内存分析工具
valgrind --tool=massif ./your_program
```

### 开发问题

#### Q1: 如何添加新模块

**步骤**：
```bash
# 1. 创建模块目录
mkdir aura-new-module

# 2. 创建 CMakeLists.txt
cat > aura-new-module/CMakeLists.txt << EOF
set(TARGET_NAME aura_new_module)
add_library(${TARGET_NAME} src/implementation.cpp)
target_include_directories(${TARGET_NAME} PUBLIC include)
EOF

# 3. 在主 CMakeLists.txt 中添加
# add_subdirectory(aura-new-module)
```

#### Q2: 如何调试 CMake 构建

**调试方法**：
```bash
# 启用详细输出
cmake --trace ..

# 查看变量
cmake -LA ..

# 生成 compile_commands.json
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
```

---

*文档版本：1.0*  
*最后更新：2026年4月30日*

如需更多帮助，请参考：
- `TECHNICAL_ARCHITECTURE.md` - 技术架构详情
- `USAGE_DOCUMENTATION.md` - 使用指南
- 项目中的 `examples/` 和 `tests/` 目录
