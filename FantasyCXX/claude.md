# FantasyCXX 项目 Claude Code 指南

## 项目概览
FantasyCXX（AuraKaleidos C++ 项目）是一个综合性 C++ 项目集合，主要专注于计算机视觉、性能监控和配置解析等领域的开发。项目使用 CMake 作为构建系统，支持跨平台编译。

## 项目结构

### 核心模块
- **aura-cv**: 计算机视觉核心模块，提供图像处理和视觉算法功能
- **aura-vision**: 视觉处理模块，包含人脸检测、手势识别、身体检测等功能
- **aura-vision-hpc**: 高性能计算视觉模块
- **aura-vision-tools**: 视觉处理工具集

### 工具模块
- **aura-perfguard**: 性能监控和防护模块
- **aura-config-parser**: 配置文件解析器（支持 INI 格式）
- **aura-utils**: 通用工具库
- **aura-lightbuffer**: 轻量级缓冲区管理

### 应用模块
- **aura-auto-driving**: 自动驾驶相关功能模块

## 构建系统

### 构建脚本使用方法
```bash
# 构建当前平台（Linux/macOS）
./build.sh

# 构建 Android arm64-v8a 版本
./build.sh -t 2 -d  # 调试版本
./build.sh -t 2 -r  # 发布版本

# 构建 Android armeabi-v7a 版本
./build.sh -t 1 -d  # 调试版本
./build.sh -t 1 -r  # 发布版本
```

### CMake 构建选项
- `BUILD_STATIC_LIB`: 构建静态库（默认：OFF）
- `BUILD_EXAMPLES`: 构建示例代码（默认：ON）
- `BUILD_TOOLS`: 构建工具（默认：ON）
- `BUILD_UNIT_TEST`: 构建单元测试（默认：ON）

### 第三方依赖
- OpenCV: 计算机视觉库
- GTest: 单元测试框架
- JSON: JSON 解析库
- Ceres: 非线性优化库
- Eigen: 线性代数库
- Boost: C++ 扩展库
- OSQP: 优化求解器

## 开发规范

### 代码风格
- 使用 `.clang-format` 进行代码格式化
- 需要符合 C++17 标准
- 遵循现有的模块组织结构

### 添加新模块
1. 在项目根目录创建新模块文件夹
2. 创建 CMakeLists.txt 文件
3. 在主 CMakeLists.txt 中添加子目录引用
4. 遵循现有的代码组织结构

### 测试方法
```bash
# 构建测试
cmake -DBUILD_UNIT_TEST=ON ..
make

# 运行测试
cd test && ./run_tests
```

## 平台支持

### 移动平台
- **Android**: arm64-v8a 和 armeabi-v7a 架构
- **iOS**: arm64 架构

### 桌面平台
- **Linux**: x86_64 架构
- **Windows**: x86_64 和 x86 架构
- **macOS**: x86_64 架构

### 嵌入式平台
- **Hexagon**: Qualcomm DSP 平台
- **Xtensa**: Cadence DSP 平台

## 配置参数

构建时的 CMake 参数：
- `TARGET_OS`: 目标操作系统
- `TARGET_ARCH`: 目标架构
- `CMAKE_BUILD_TYPE`: 构建类型（debug/release）
- `PRODUCTION`: 产品名称
- `SOC_VENDOR`: SoC 供应商

## Claude Code 使用提示

### 常见任务
1. **项目构建**: 使用 `./build.sh` 或手动 CMake 命令
2. **添加新模块**: 遵循现有目录中的模块结构
3. **代码格式化**: 使用提供的 `.clang-format` 配置
4. **测试**: 启用 `BUILD_UNIT_TEST` 并从 test 目录运行测试

### 关键文件
- `CMakeLists.txt`: 主构建配置文件
- `build.sh`: 构建自动化脚本
- `.clang-format`: 代码风格配置文件
- `README.md`: 项目文档

### 模块开发
在处理单个模块时：
- 检查模块的 CMakeLists.txt 了解依赖关系
- 查看 `examples/` 目录中的现有示例
- 确保与目标平台的兼容性
- 为新功能添加适当的单元测试

## 贡献规范

1. 代码必须遵循项目编码标准
2. 新功能需要添加相应的单元测试
3. 更新相关文档
4. 提交前在目标平台上测试通过

## 近期开发重点
基于 git 历史记录，近期工作主要集中在：
- 文档完善（README.md 更新）
- 添加新的演示和示例
- CUDA 演示集成
- 代码注释和文档补充
- 第三方库集成

## 故障排除

### 构建问题
- 确保安装了 CMake 3.10 或更高版本
- 检查第三方依赖是否可用
- 验证目标平台工具链配置是否正确

### 平台特定问题
- Android 构建需要配置 NDK
- 交叉编译可能需要额外的工具链配置
- 查看模块文档中的平台特定要求