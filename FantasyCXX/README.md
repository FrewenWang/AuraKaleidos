# C++ 万花筒 (AuraKaleidos C++ Project)

这是一个综合性C++项目集合，包含多个功能模块，主要用于计算机视觉、性能监控、配置解析等领域的开发。

## 项目结构

本项目采用模块化设计，主要包含以下子模块：

### 核心模块

- **aura-cv**: 计算机视觉核心模块，提供图像处理和视觉算法功能
- **aura-vision**: 视觉处理模块，包含人脸检测、手势识别、身体检测等功能
- **aura-vision-hpc**: 高性能计算视觉模块
- **aura-vision-tools**: 视觉处理工具集

### 工具模块

- **aura-perfguard**: 性能监控和防护模块
- **aura-config-parser**: 配置文件解析器（支持INI格式）
- **aura-utils**: 通用工具库
- **aura-lightbuffer**: 轻量级缓冲区管理

### 应用模块

- **aura-auto-driving**: 自动驾驶相关功能模块

## 构建系统

本项目使用CMake作为构建系统，支持跨平台编译。

### 构建要求

- CMake 3.10 或更高版本
- C++17 兼容编译器
- 支持的操作系统：Linux、macOS、Windows、Android

### 构建选项

主要构建选项包括：

- `BUILD_STATIC_LIB`: 构建静态库（默认OFF）
- `BUILD_EXAMPLES`: 构建示例代码（默认ON）
- `BUILD_TOOLS`: 构建工具（默认ON）
- `BUILD_UNIT_TEST`: 构建单元测试（默认ON）

### 第三方依赖

项目支持集成以下第三方库：

- OpenCV: 计算机视觉库
- GTest: 单元测试框架
- JSON: JSON解析库
- Ceres: 非线性优化库
- Eigen: 线性代数库
- Boost: C++扩展库
- OSQP: 优化求解器

## 快速开始

### 使用构建脚本

项目提供了便捷的构建脚本：

```bash
# 构建当前平台（Linux/macOS）
./build.sh

# 构建Android arm64-v8a版本
./build.sh -t 2 -d  # 调试版本
./build.sh -t 2 -r  # 发布版本

# 构建Android armeabi-v7a版本
./build.sh -t 1 -d  # 调试版本
./build.sh -t 1 -r  # 发布版本
```

### 手动构建

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置项目
cmake -DCMAKE_BUILD_TYPE=Release ..

# 编译
make -j$(nproc)

# 安装
make install
```

## 示例代码

项目包含丰富的示例代码，位于 `examples/` 目录：

- `01.demo_sizeof.cpp`: C++数据类型大小演示
- `ini_parser/`: 配置文件解析示例
- `object_detection/`: 目标检测示例

## 平台支持

### 移动端平台

- **Android**: 支持arm64-v8a和armeabi-v7a架构
- **iOS**: 支持arm64架构

### 桌面平台

- **Linux**: x86_64架构
- **Windows**: x86_64和x86架构
- **macOS**: x86_64架构

### 嵌入式平台

- **Hexagon**: Qualcomm DSP平台
- **Xtensa**: Cadence DSP平台

## 配置参数

构建时可通过CMake参数自定义配置：

- `TARGET_OS`: 目标操作系统
- `TARGET_ARCH`: 目标架构
- `CMAKE_BUILD_TYPE`: 构建类型（debug/release）
- `PRODUCTION`: 产品名称
- `SOC_VENDOR`: SoC供应商

## 开发指南

### 代码风格

项目使用.clang-format文件定义代码风格，建议使用ClangFormat进行代码格式化。

### 添加新模块

1. 在项目根目录创建新模块文件夹
2. 创建CMakeLists.txt文件
3. 在主CMakeLists.txt中添加子目录引用
4. 遵循现有的代码组织结构

## 测试

项目支持单元测试，可通过以下方式运行：

```bash
# 构建测试
cmake -DBUILD_UNIT_TEST=ON ..
make

# 运行测试
cd test && ./run_tests
```

## 许可证

本项目代码遵循相应的开源许可证，具体请参考各模块的LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request。在提交代码前，请确保：

1. 代码符合项目的编码规范
2. 添加了相应的单元测试
3. 更新了相关文档
4. 在目标平台上测试通过

## 联系方式

如有问题或建议，请通过项目Issue系统提交。
