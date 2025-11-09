# aura示例

本教程描述了如何使用aura构建一个C++可执行程序，教程主要有以下部分：
- 介绍aura库的获取方式与目录结构
- 以高斯滤波为例，介绍如何使用aura库
- 使用cmake构建可执行程序
- 编译android、linux、windows下的可执行程序

本教程的目录结构如下：
```
.
├── aura            # aura库目录
│   └── 3rdparty    # 存放三方库
│   └── cmake       # 存放cmake相关文件
│   └── include     # 存放aura头文件
|   └── lib         # 存放aura库文件
|   └── sample      # 存放aura使用示例
├── build           # 空目录，存放cmake构建文件
├── data
│   └── comm        # 测试图像
├── main.cpp        # 示例程序
└── CMakeLists.txt  # aura使用示例
```

## 1. aura库获取

aura库可在[预编译库发版页](https://git.n.xiaomi.com/mi-camera-algorithm-release/aura2.0/-/releases)中下载，目前aura提供的预编译库支持的平台及配置如下：

| 平台 | 配置 | 说明 |
| :---: | :---: | :---: |
| android | {static &#124; share} &lt;arm64-v8a&gt; &lt;release&gt; \[asan &#124; hwasan\] {android &#124; qcom &#124; nnlite} | qcom支持hexagon，nnlite仅包含NN模块 |
| hexagon | &lt;share&gt; {v68 &#124; v69 &#124; v73 &#124; v75} | - |
| linux   | {static &#124; share} &lt;x64&gt; &lt;release&gt; | - |
| windows | {static &#124; share} &lt;x64&gt; &lt;release&gt; | - |

<!-- 用户可以根据自身需求选择，选择下载适合的预编译库，如果预编译库中的配置无法满足用户需求，用户可以参考[aura编译方法](...)自行编译aura库。 -->

aura 发版页中不仅包含不同版本的预编译 aura 库，还提供了方便进行 aura 库管理的脚本`get_aura2.py`和相应的版本配置文件`aura2_versions.json`（可在[get_aura2](http://10.221.116.29:5000/aura2.zip)中下载解压得到）。当用户需要使用 aura 库时，可以通过修改`aura2_version.json`中的配置， 运行脚本`get_aura2.py`来获取或更新 aura 库。

下面示例将下载并解压`aura_2.0.0_qcom_ndkr26c_arm64-v8a_static`和`aura_2.0.0_hexagon_v68`两种库到`get_aura2.py`所在路径。
```bash
# aura2_version.json
# {
#     "aura_2.0.0_qcom_ndkr26c_arm64-v8a_share"  : 0,
#     "aura_2.0.0_qcom_ndkr26c_arm64-v8a_static" : 1,
#     "aura_2.0.0_hexagon_v68"                   : 1,
#     "aura_2.0.0_hexagon_v69"                   : 0,
#}

python get_aura2.py aura2_version.json
```

## 2. 编写main.cpp调用aura高斯滤波函数

aura库的调用包括以下步骤：
- 根据所需功能，引入aura头文件，本示例调用算子模块中的高斯滤波函数，需要引入`aura/ops/filter.h`头文件
- 设定上下文配置参数，创建aura上下文，如指定线程池、OpenCL配置等
- 基于aura上下文创建aura::Mat输入输出对象，调用aura库函数

### 2.1 引入aura filter头文件

```cpp
#include "aura/ops/filter.h"
```

### 2.2 创建上下文

```cpp
std::shared_ptr<aura::Context> CreateContext()
{
    aura::Config config;                                            // 上下文配置参数
    config.SetWorkerPool("AuraSample", aura::CpuAffinity::BIG, aura::CpuAffinity::LITTLE);  // 设置线程池

#if defined(AURA_ENABLE_OPENCL)
    config.SetCLConf(DT_TRUE, "/data/local/tmp", "aura_unit_test"); // 设置OpenCL配置
#endif

#if defined(AURA_ENABLE_HEXAGON)
    config.SetHexagonConf(DT_TRUE, DT_TRUE, "aura_hexagon");        // 设置Hexagon配置
#endif

#if defined(AURA_ENABLE_XTENSA)
    config.SetXtensaConf(DT_TRUE, "aura_xtensa_pil.so", XtensaPriorityLevel::PRIORITY_HIGH);  // 设置Xtensa配置
#endif

    std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(config);           // 创建上下文
    if (ctx->Initialize() != aura::Status::OK)                      // 初始化上下文
    {
        AURA_LOGE(ctx, "AURA_GAUSSIAN_SAMPLE", "aura::Context::Initialize() failed\n");     // 打印错误信息
        return DT_NULL;
    }

    return ctx;
}
```

### 2.3 调用aura库函数（高斯滤波）

```cpp
aura::Status GaussianBlur(aura::Context* ctx)
{
    aura::Sizes3 size = {487, 487, 1};              // 输入输出图像大小

    aura::Mat src(ctx, aura::ElemType::U8, size);   // 输入图像
    aura::Mat dst(ctx, aura::ElemType::U8, size);   // 输出图像
    if (!(src.IsValid() && dst.IsValid()))          // 验证输入输出图像是否有效
    {
        AURA_LOGE(ctx, "AURA_GAUSSIAN_SAMPLE", "failed to create aura::Mat"); // 打印错误信息
        return aura::Status::ERROR;
    }

    if (src.Load("data/comm/cameraman_487x487.gray") != aura::Status::OK)   // 加载输入图像
    {
        AURA_LOGE(ctx, "AURA_GAUSSIAN_SAMPLE", "failed to load data to mat: %s\n", ctx->GetLogger()->GetErrorString().c_str()); // 打印错误信息
        return aura::Status::ERROR;
    }

    /* 注意：
     *  - 调用aura高斯滤波函数，aura::OpTarget::None()可根据需求替换为Neon/Opencl/Hvx/Vdsp
     *  - 调用hexagon方法前，需要将libaura_hexagon_skel.so拷贝到目标设备，并设置环境变量"export ADSP_LIBRARY_PATH=${ADSP_LIBRARY_PATH};YOUR_SKEL_PATH"
     *  - 调用xtensa方法前，需要将aura_xtensa_pil.so拷贝到目标设备，并与可执行文件存放在同一路径下
     */
    if (IGaussian(ctx, src, dst, 3, 1.0f, aura::BorderType::REFLECT_101, aura::Scalar(0, 0, 0, 0), aura::OpTarget::None()) != aura::Status::OK)
    {
        AURA_LOGE(ctx, "AURA_GAUSSIAN_SAMPLE", "failed to run IGaussian: %s\n", ctx->GetLogger()->GetErrorString().c_str()); // 打印错误信息
        return aura::Status::ERROR;
    }

    dst.Dump("./gaussian_result.raw");              // 保存输出图像

    return aura::Status::OK;
}
```

### 2.4 编写main函数

```cpp
int main()
{
    std::shared_ptr<aura::Context> ctx = CreateContext();
    if (DT_NULL == ctx)
    {
        return -1;
    }

    if (GaussianBlur(ctx.get()) != aura::Status::OK)
    {
        AURA_LOGE(ctx, "AURA_GAUSSIAN_SAMPLE", "failed to run GaussianBlur\n");
        return -1;
    }

    return 0;
}
```

## 3. 编写CMakeLists.txt构建可执行程序

aura使用cmake构建可执行程序，在cmake中推荐使用find_package命令查找并设置aura库，示例如下：
```cmake
set(Aura_DIR "YOUR_AURA_INSTALL_PATH/cmake")

find_package(Aura REQUIRED)
target_link_libraries(YOUR_TARGET_NAME ${Aura_LIBS})
```
通过find_package命令查找aura库之前，需要对aura库的搜索路径`Aura_DIR`进行设置，查找成功之后，aura库的相关变量会暴露出来，用户可以通过这些变量来配置自己的可执行程序，aura库的相关变量如下：

| 变量名 | 说明 |
| :---: | :---: |
| Aura_FOUND | 查找aura库是否成功 |
| Aura_INCLUDE_DIRS | aura头文件目录列表 |
| Aura_LIBS | aura库列表 |
| Aura_ANDROID_ABI | android-abi，仅android平台有效 |
| Aura_ANDROID_PLATFORM | android-platform，仅android平台有效 |
| Aura_ENABLE_ARM82 | 启用neon-fp，仅android平台有效 |
| Aura_ENABLE_HEXAGON | 启用hexagon模块，仅android（qualcomm）平台有效 |
| Aura_ENABLE_OPENCL | 启用opencl模块，仅android平台有效 |
| Aura_SHARED_LIBRARY | aura库是否为动态库 |
| Aura_INSTALL_PATH | aura库安装路径 |

示例CMakeListe.txt内容如下：
```cmake
cmake_minimum_required(VERSION 3.2)

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()

project(sample)

# 需要在cmake命令行中通过"-DAura_DIR=YOUR_AURA_INSTALL_PATH/cmake"指定aura库的路径
find_package(Aura REQUIRED)
if(NOT Aura_FOUND)
    message(FATAL_ERROR "failed to find aura")
endif()

add_executable(sample_gaussian main.cpp)
target_link_libraries(sample_gaussian ${Aura_LIBS})

install(TARGETS sample_gaussian DESTINATION .)
```

## 4. 编译android、linux、windows下的可执行程序

编译可执行程序时，首先需要确保目标程序架构与aura库的架构相匹配。使用cmake构建可执行程序前，需要先切换到build目录。

### 4.1 android平台

android目标平台需要使用ndk进行交叉编译，用户需要提前指定ndk的安装路径，示例如下：
```shell
cmake -DAura_DIR=../aura/cmake \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-23 \
      -DANDROID_STL=c++_shared \
      .. && make -j8 install
```
在上述命令中，Aura_DIR指定了aura库的安装路径，ANDROID_NDK_ROOT是ndk的安装路径。
注意，ANDROID_ABI和ANDROID_PLATFORM需要和aura库配置的参数保持一致，否则会触发编译错误，示例命令将编译出arm64-v8a架构下的可执行程序。

### 4.2 linux平台

linux平台可使用gnu或clang进行编译，示例如下：
```shell
cmake -DAura_DIR=../aura/cmake .. && make -j8 install
```
注意，当编译x86程序时，需要在cmake命令行添加参数`-DCMAKE_C_FLAGS=-m32 -DCMAKE_CXX_FLAGS=-m32`，示例命令将编译出x64架构下的可执行程序。

### 4.3 windows平台

windows平台使用visual studio进行编译，示例如下：
```shell
cmake -DAura_DIR=../aura/cmake -A x64 -G "Visual Studio 16 2019".. && cmake --build . --target install --config release -- /m:8
```
注意，在windows平台下进行编译时，需要保证--config的参数与aura库的编译参数一致，否则会触发编译错误。

## 5. 更多内容

更多使用方式，可以参考aura/sample目录下的示例。