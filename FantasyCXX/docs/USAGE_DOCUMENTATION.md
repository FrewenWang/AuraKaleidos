# FantasyCXX 使用文档

## 目录

1. [快速开始](#快速开始)
2. [模块使用指南](#模块使用指南)
3. [构建配置详解](#构建配置详解)
4. [示例程序](#示例程序)
5. [API 参考](#api-参考)
6. [最佳实践](#最佳实践)
7. [故障排除](#故障排除)

## 快速开始

### 环境准备

#### 系统要求
- **操作系统**：Linux、macOS、Windows
- **CMake**：3.10 或更高版本
- **C++ 编译器**：支持 C++17 标准
- **磁盘空间**：至少 2GB 可用空间

#### 安装依赖

**Ubuntu/Debian**：
```bash
sudo apt update
sudo apt install -y cmake build-essential git
```

**macOS**：
```bash
# 使用 Homebrew
brew install cmake
```

**Windows**：
- 安装 [CMake](https://cmake.org/download/)
- 安装 Visual Studio 2019 或更高版本

### 获取代码

```bash
git clone <repository_url>
cd FantasyCXX
```

### 快速构建

**Linux/macOS**：
```bash
# 使用构建脚本（推荐）
./build.sh

# 或手动构建
mkdir build && cd build
cmake ..
make -j$(nproc)
```

**Windows**：
```bash
mkdir build
cd build
cmake -G "Visual Studio 16 2019" -A x64 ..
cmake --build . --config Release
```

## 模块使用指南

### aura-cv 计算机视觉模块

#### 功能概述
aura-cv 提供基础的计算机视觉算法，包括图像处理、特征提取、目标检测等功能。

#### 基本使用

**包含头文件**：
```cpp
#include "aura/aura_cv/aura_cv.h"
```

**初始化模块**：
```cpp
// 初始化 AuraCV 环境
AuraCV::Init();

// 创建图像处理上下文
auto context = AuraCV::CreateContext();
```

**图像处理示例**：
```cpp
// 加载图像
auto image = AuraCV::LoadImage("path/to/image.jpg");

// 图像灰度化
auto gray_image = AuraCV::CvtColor(image, AuraCV::COLOR_BGR2GRAY);

// 边缘检测
auto edges = AuraCV::Canny(gray_image, 50, 150);
```

#### 高级特性

**硬件加速**：
```cpp
// 启用 OpenCL 加速
AuraCV::Config config;
config.enable_opencl = true;
AuraCV::Init(config);
```

**自定义算子**：
```cpp
class CustomFilter : public AuraCV::Operator {
public:
    void Execute(const AuraCV::Tensor& input, AuraCV::Tensor& output) override {
        // 自定义处理逻辑
    }
};

// 注册自定义算子
AuraCV::RegisterOperator("custom_filter", std::make_shared<CustomFilter>());
```

### aura-config-parser 配置解析模块

#### 功能概述
轻量级配置解析器，支持 INI 格式和 Protocol Buffer 文本格式。

#### 基本使用

**INI 文件解析**：
```cpp
#include "aura/config_parser/config_parser.h"

// 解析配置文件
AuraConfig::Parser parser;
parser.ParseFile("config.ini");

// 读取配置值
std::string value = parser.GetValue("section", "key");
int port = parser.GetIntValue("network", "port", 8080);  // 默认值 8080
```

**示例配置文件 (config.ini)**：
```ini
[network]
host = localhost
port = 8080

database
host = 127.0.0.1
port = 3306
name = test_db
```

**Protocol Buffer 文本格式**：
```cpp
// 解析 PB 文本格式
std::string pb_text = R"(
    name: "test"
    value: 123
    items: ["item1", "item2"]
)";

auto config = AuraConfig::ParseFromText(pb_text);
```

#### 配置热加载

```cpp
// 启用配置热加载
parser.EnableHotReload(true);

// 监听配置变化
parser.OnConfigChanged([](const std::string& key, const std::string& value) {
    std::cout << "Config changed: " << key << " = " << value << std::endl;
});
```

### aura-vision 视觉处理模块

#### 功能概述
高级视觉处理功能，包括人脸检测、手势识别、姿态估计等。

#### 人脸检测示例

```cpp
#include "aura/vision/face_detector.h"

// 创建人脸检测器
auto detector = AuraVision::CreateFaceDetector();

// 加载模型
detector->LoadModel("models/face_detection.caffe");

// 检测人脸
cv::Mat image = cv::imread("test.jpg");
auto faces = detector->Detect(image);

// 处理检测结果
for (const auto& face : faces) {
    std::cout << "Face detected at: " << face.bbox << std::endl;
    std::cout << "Confidence: " << face.confidence << std::endl;
}
```

#### 手势识别

```cpp
#include "aura/vision/gesture_recognizer.h"

// 创建手势识别器
auto recognizer = AuraVision::CreateGestureRecognizer();

// 实时手势识别
cv::VideoCapture cap(0);
while (true) {
    cv::Mat frame;
    cap >> frame;
    
    auto gestures = recognizer->Recognize(frame);
    for (const auto& gesture : gestures) {
        std::cout << "Gesture: " << gesture.type << std::endl;
    }
}
```

### aura-perfguard 性能监控模块

#### 功能概述
系统性能监控和防护，用于监控算法性能和资源使用。

#### 基本使用

```cpp
#include "aura/perfguard/performance_monitor.h"

// 创建性能监控器
auto monitor = AuraPerfGuard::CreateMonitor();

// 开始监控特定函数
AURA_PERF_SCOPE("image_processing");

// 或者手动记录
monitor->StartTimer("algorithm_time");
// ... 执行算法 ...
monitor->StopTimer("algorithm_time");

// 获取性能报告
auto report = monitor->GenerateReport();
std::cout << report << std::endl;
```

#### 内存监控

```cpp
// 启用内存监控
monitor->EnableMemoryMonitoring(true);

// 设置内存阈值
monitor->SetMemoryThreshold(1024 * 1024 * 1024);  // 1GB

// 监听内存警告
monitor->OnMemoryWarning([]() {
    std::cout << "Memory usage exceeded threshold!" << std::endl;
});
```

### aura-auto-driving 自动驾驶模块

#### 功能概述
自动驾驶相关算法，包括目标跟踪、轨迹预测等。

#### 目标跟踪

```cpp
#include "aura/auto_driving/kalman_filter.h"

// 创建 Kalman 滤波器
auto filter = AuraAutoDriving::CreateKalmanCTRVFilter();

// 初始化状态
AuraAutoDriving::State state;
state.position = {x, y};
state.velocity = {vx, vy};
filter->Initialize(state);

// 预测和更新
auto predicted = filter->Predict(dt);
filter->Update(measurement);
```

#### 多目标匹配

```cpp
#include "aura/auto_driving/hungarian_matcher.h"

// 创建匹配器
auto matcher = AuraAutoDriving::CreateGatedHungarianMatcher();

// 执行匹配
std::vector<Detection> detections = GetDetections();
std::vector<Track> tracks = GetTracks();

auto matches = matcher->Match(detections, tracks);

// 处理匹配结果
for (const auto& match : matches) {
    std::cout << "Detection " << match.detection_id 
              << " matched with Track " << match.track_id << std::endl;
}
```

## 构建配置详解

### CMake 配置选项

#### 构建类型选项

```bash
# 调试版本
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 发布版本
cmake -DCMAKE_BUILD_TYPE=Release ..

# 带调试信息的发布版本
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

#### 功能模块选项

```bash
# 构建静态库
cmake -DBUILD_STATIC_LIB=ON ..

# 禁用示例构建
cmake -DBUILD_EXAMPLES=OFF ..

# 禁用工具构建
cmake -DBUILD_TOOLS=OFF ..

# 启用单元测试
cmake -DBUILD_UNIT_TEST=ON ..
```

#### 第三方库选项

```bash
# 启用 OpenCV
cmake -DBUILD_OPENCV=ON ..

# 启用 Ceres Solver
cmake -DBUILD_CERES=ON ..

# 启用 Eigen
cmake -DBUILD_EIGEN=ON ..

# 启用 GTest
cmake -DBUILD_GTEST_LIB=ON ..
```

### 平台特定配置

#### Android 构建

```bash
# 设置 NDK 路径
export ANDROID_NDK_ROOT=/path/to/android-ndk

# 构建 arm64-v8a
./build.sh -t 2 -d

# 构建 armeabi-v7a
./build.sh -t 1 -d
```

#### iOS 构建

```bash
# 使用 iOS 工具链
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake \
      -DIOS_PLATFORM=OS \
      -DIOS_ARCH=arm64 ..
```

#### Hexagon DSP 构建

```bash
# Hexagon v75
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/hexagon_toolchain.cmake \
      -DAURA_HEXAGON_ARCH=v75 ..
```

## 示例程序

### 基础示例

#### 数据类型大小演示

```cpp
// examples/01.demo_sizeof.cpp
#include <cstdint>
#include <cstdio>

int main() {
    printf("sizeof(char)==%lu\n", sizeof(char));
    printf("sizeof(int)==%lu\n", sizeof(int));
    printf("sizeof(double)==%lu\n", sizeof(double));
    
    // 结构体大小演示
    struct Hello {
        int id;
        short age;
        short agent;
        double score;
    };
    
    printf("sizeof(Hello)==%lu\n", sizeof(Hello));
    return 0;
}
```

### 配置解析示例

```cpp
// examples/ini_parser/demo.cpp
#include "aura/config_parser/config_parser.h"

int main() {
    AuraConfig::Parser parser;
    
    // 解析配置文件
    if (!parser.ParseFile("config.ini")) {
        std::cerr << "Failed to parse config file" << std::endl;
        return -1;
    }
    
    // 读取配置值
    std::string host = parser.GetValue("network", "host");
    int port = parser.GetIntValue("network", "port", 8080);
    
    std::cout << "Server: " << host << ":" << port << std::endl;
    return 0;
}
```

### 目标检测示例

```cpp
// examples/object_detection/demo.cpp
#include "aura/vision/object_detector.h"
#include "opencv2/opencv.hpp"

int main() {
    // 初始化检测器
    auto detector = AuraVision::CreateObjectDetector();
    detector->LoadModel("models/ssd_mobilenet.caffe");
    
    // 加载图像
    cv::Mat image = cv::imread("test.jpg");
    
    // 执行检测
    auto detections = detector->Detect(image);
    
    // 绘制结果
    for (const auto& det : detections) {
        cv::rectangle(image, det.bbox, cv::Scalar(0, 255, 0));
        cv::putText(image, det.class_name, det.bbox.tl(), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0));
    }
    
    cv::imwrite("result.jpg", image);
    return 0;
}
```

## API 参考

### 核心 API

#### AuraCV 核心接口

```cpp
namespace AuraCV {
    // 初始化
    bool Init(const Config& config = Config());
    void Shutdown();
    
    // 上下文管理
    std::shared_ptr<Context> CreateContext();
    
    // 图像处理
    Image LoadImage(const std::string& path);
    Image CvtColor(const Image& image, ColorCode code);
    Image Canny(const Image& image, double threshold1, double threshold2);
}
```

#### 配置解析 API

```cpp
namespace AuraConfig {
    class Parser {
    public:
        bool ParseFile(const std::string& filename);
        bool ParseString(const std::string& content);
        
        std::string GetValue(const std::string& section, const std::string& key);
        int GetIntValue(const std::string& section, const std::string& key, int default_value = 0);
        double GetDoubleValue(const std::string& section, const std::string& key, double default_value = 0.0);
        bool GetBoolValue(const std::string& section, const std::string& key, bool default_value = false);
        
        void EnableHotReload(bool enable);
        void OnConfigChanged(std::function<void(const std::string&, const std::string&)> callback);
    };
}
```

## 最佳实践

### 代码规范

#### 命名约定
- **类名**：使用 PascalCase，如 `FaceDetector`
- **函数名**：使用 camelCase，如 `loadImage`
- **变量名**：使用 camelCase，如 `imagePath`
- **常量**：使用 UPPER_CASE，如 `MAX_WIDTH`

#### 内存管理

```cpp
// 推荐：使用智能指针
std::shared_ptr<Detector> detector = CreateDetector();

// 避免：裸指针管理
Detector* detector = new Detector();
// ... 容易忘记 delete
delete detector;
```

#### 错误处理

```cpp
// 推荐：异常处理
try {
    auto result = detector->Detect(image);
    // 处理结果
} catch (const AuraException& e) {
    std::cerr << "Detection failed: " << e.what() << std::endl;
}

// 推荐：错误码检查
auto result = detector->Detect(image);
if (!result.success) {
    std::cerr << "Detection failed: " << result.error_message << std::endl;
}
```

### 性能优化

#### 图像处理优化

```cpp
// 推荐：复用图像缓冲区
cv::Mat buffer;
for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    
    // 预分配缓冲区
    if (buffer.empty() || buffer.size() != image.size()) {
        buffer = cv::Mat(image.size(), image.type());
    }
    
    // 使用缓冲区进行处理
    cv::cvtColor(image, buffer, cv::COLOR_BGR2GRAY);
    // ... 其他处理
}
```

#### 并行处理

```cpp
// 推荐：多线程处理
std::vector<std::future<Result>> futures;
for (const auto& task : tasks) {
    futures.push_back(std::async(std::launch::async, [&task]() {
        return ProcessTask(task);
    }));
}

// 等待所有任务完成
for (auto& future : futures) {
    auto result = future.get();
    // 处理结果
}
```

## 故障排除

### 常见构建问题

#### CMake 找不到依赖

**问题**：`Could NOT find OpenCV`

**解决方案**：
```bash
# 设置 OpenCV 路径
cmake -DOpenCV_DIR=/path/to/opencv/build ..

# 或禁用 OpenCV
cmake -DBUILD_OPENCV=OFF ..
```

#### 编译器错误

**问题**：C++17 特性不支持

**解决方案**：
```bash
# 检查编译器版本
g++ --version

# 更新编译器或设置标准
cmake -DCMAKE_CXX_STANDARD=14 ..
```

### 运行时问题

#### 动态库加载失败

**问题**：`error while loading shared libraries`

**解决方案**：
```bash
# 设置库路径
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH

# 或链接静态库
cmake -DBUILD_STATIC_LIB=ON ..
```

#### 内存不足

**问题**：处理大图像时内存不足

**解决方案**：
```cpp
// 使用图像金字塔
auto small_image = AuraCV::Resize(image, {640, 480});
auto result = detector->Detect(small_image);

// 或使用流式处理
AuraCV::StreamProcessor processor;
processor.ProcessInChunks(image, chunk_size);
```

### 性能问题

#### 算法执行慢

**诊断步骤**：
```cpp
// 使用性能监控
AURA_PERF_SCOPE("detection_pipeline");

// 检查硬件加速是否启用
if (!AuraCV::IsOpenCLEnabled()) {
    std::cout << "OpenCL not enabled, performance may be suboptimal" << std::endl;
}
```

**优化建议**：
- 启用硬件加速
- 使用合适的图像分辨率
- 批量处理减少开销
- 缓存中间结果

---

*文档版本：1.0*  
*最后更新：2026年4月30日*

如需更多帮助，请参考项目中的 `examples/` 目录和 `test/` 目录中的示例代码。