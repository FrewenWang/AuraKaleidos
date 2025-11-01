# NN

## 目录

- [1. 简介](#1-简介)
- [2. 快速上手](#2-快速上手)
  - [2.1 编写一个简单的InceptionV3示例](#21-编写一个简单的inceptionv3示例)
    - [2.1.1 创建上下文并获取推理引擎](#211-创建上下文并获取推理引擎)
    - [2.1.2 创建执行器](#212-创建执行器)
    - [2.1.3 执行推理](#213-执行推理)
    - [2.1.4 推理验证](#214-推理验证)
  - [2.2 使用CMake编译](#22-使用cmake编译)
  - [2.3 运行示例](#23-运行示例)
- [3. 常见问题](#3-常见问题)

# 1. 简介

NN 是 AURA 中的神经网络模块，它针对移动端推理进行设计，将 MNN、NP、QNN、SNPE、XNN 等推理框架进行优化整合并封装成统一的接口，支持 CPU、GPU 和 NPU 等多种后端。

在不同的推理框架中，NN 对后端的支持情况有所区别，具体情况如下：
| Framework |   CPU   |   GPU   |   NPU   |
|:--------: | :-----: | :-----: | :-----: |
|    MNN    | &check; | &check; | &cross; |
|    NP     | &cross; | &cross; | &check; |
|    QNN    | &cross; | &check; | &check; |
|    SNPE   | &cross; | &cross; | &check; |
|    XNN    | &cross; | &cross; | &check; |

# 2. 快速上手

本文通过一个简单的 InceptionV3 示例来介绍如何使用 NN 模块，演示 NN 在 MNN、NP、QNN、SNPE、XNN 等不同框架下实现高效推理的过程。

## 2.1 编写一个简单的InceptionV3示例

### 2.1.1 创建上下文并获取推理引擎

NNEngine 是 NN 的大管家，负责创建不同的 NNExecutor。在 AURA 中，NNEngine 通过上下文进行管理，使用 NNEngine 之前需要创建并初始化一个上下文。当 NN 模块被启用后，NNEngine 会在上下文初始化时被创建。

在上下文的默认配置中 NN 模块是被禁用的，因此在创建上下文之前，需要调用上下文的 `SetNNConf` 方法来启用 NN 模块。
```cpp
std::shared_ptr<aura::Context> CreateNNContext()
{
    aura::Config config;
    config.SetNNConf(MI_TRUE);                                      // 使能 NN 模块

    std::shared_ptr<aura::Context> ctx = std::make_shared<aura::Context>(config);   // 创建上下文对象
    if (MI_NULL == ctx)
    {
        fprintf(stderr, "failed to create context\n");
        return MI_NULL;
    }

    if (ctx->Initialize() != aura::Status::OK)                      // 初始化上下文对象
    {
        fprintf(stderr, "failed to initialize context\n");
        return MI_NULL;
    }

    return ctx;
}
```
上下文创建完成后，可以通过调用上下文的 `GetNNEngine` 方法来获取 NNEngine。
```cpp
aura::NNEngine *nn_engine = ctx->GetNNEngine();                     // 获取推理引擎
if (!nn_engine)
{
    AURA_LOGE(ctx, NN_TAG, "failed to get nn engine: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}
```

### 2.1.2 创建执行器

NNEngine 提供了 `CreateNNExecutor` 方法来创建执行器，创建执行器时需要传入信息、模型密钥和执行器配置信息，其中模型信息可以通过模型文件和模型数据两种方式来提供，示例展示通过模型文件创建执行器的方法。

定义一个模型信息结构体 `ModelInfo` 来保存模型文件路径、测试数据路径、模型密钥以及模型输入节点列表和输出节点列表。其中模型文件路径、密钥用来创建执行器，而输入/输出节点和测试数据则在后续用于执行推理。
```cpp
struct ModelInfo                                                    // 模型相关的配置信息
{
    std::string model_file;                                         // 模型文件路径
    std::string key;                                                // 模型密钥，用于解密，与转换 minn 模型时设置的加密密钥一致
    std::vector<std::string> input_node;                            // 模型输入节点列表，可通过 netron 读取原始模型获取，也可通过调用执行器的 GetInputs 方法获取
    std::vector<std::string> output_node;                           // 模型输出节点列表，可通过 netron 读取原始模型获取，也可通过调用执行器的 GetOutputs 方法获取
    std::string input_file;                                         // 模型测试数据路径
};
```
接下来创建一个映射结构 `model_list`用来保存模型的详细信息，模型分为以下两种：

- 单模型模式，只包含一个minn模型，成员如下：
  - 框架格式，例如：minn_mnn
  - 模型路径，例如：/data/local/tmp/aura/data/nn/mnn/inception_v3_mnn_gpu_v271.minn

- 容器模型模式，将多个minn模型打包，成员如下：
  - 框架格式，例如：minb_snpe
  - 模型路径，例如：/data/local/tmp/aura/minb/inception_v3_snpev224_qnnv224.minb
  - 模型名称，例如：inception_v3_snpe_npu_v224
```cpp
std::unordered_map<std::string, ModelFileInfo> model_list =
{
    {
        "minn_mnn",  {"/data/local/tmp/aura/data/nn/mnn/inception_v3_mnn_gpu_v271.minn"}
    },
    {
        "minn_np",   {"/data/local/tmp/aura/data/nn/np/inception_v3_np_npu_v7.minn"},
    },
    {
        "minn_qnn",  {"/data/local/tmp/aura/data/nn/qnn/inception_v3_qnn_npu_v213.minn"},
    },
    {
        "minn_snpe", {"/data/local/tmp/aura/data/nn/snpe/inception_v3_snpe_npu_v213.minn"},
    },
    {
        "minn_xnn",  {"/data/local/tmp/aura/data/nn/xnn/inception_v3_xnn_npu_v051.minn"},
    },
    {
        "minb_snpe", {"/data/local/tmp/aura/data/nn/minb/inception_v3_snpev224_qnnv224.minb", "inception_v3_snpe_npu_v224"},
    },
    {
        "minb_qnn",  {"/data/local/tmp/aura/data/nn/minb/inception_v3_snpev224_qnnv224.minb", "inception_v3_qnn_npu_v224"},
    }
};

ModelInfo model;                                                                // 定义模型信息结构体
model.model_file  = model_list.at("minn_mnn").file_path;                        // 设置模型文件路径
model.key         = "abcdefg";                                                  // 设置模型密钥
model.input_node  = {"input"};                                                  // 设置输入节点名称
model.output_node = {"InceptionV3/Predictions/Reshape_1"};                      // 设置输出节点名称
model.input_file  = "/data/local/tmp/aura/data/nn/trash_1x299x299x3_f32.bin";   // 设置测试数据路径
```
创建执行器时，可以对执行器的部分参数进行配置，本例仅为演示配置参数用法，详细的配置信息可参阅 `NNConfig`，如果没有配置需求可以不做配置，直接使用默认参数。
```cpp
aura::NNConfig config;                                              // 创建执行器配置
config["perf_level"] = "PERF_NORMAL";                               // 设置性能等级为普通模式
```
准备好执行器的接口参数后，调用 NNEngine 的 `CreateNNExecutor` 方法来创建执行器，创建完成后，调用执行器的 `Initialize` 方法来对其进行初始化。
```cpp
std::shared_ptr<aura::NNExecutor> nn_executor = nn_engine->CreateNNExecutor(model.model_file, model.key, config);   // 创建执行器对象
if (!nn_executor)
{
    AURA_LOGE(ctx, NN_TAG, "failed to create nn executor: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}

if (nn_executor->Initialize() != aura::Status::OK)                  // 初始化执行器对象
{
    AURA_LOGE(ctx, NN_TAG, "failed to initialize nn executor: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}
```
对于容器模式，需要额外添加模型名字的参数，调用 NNEngine 的 `CreateNNExecutor` 方法来创建执行器，其他流程同上。
```cpp
std::shared_ptr<aura::NNExecutor> nn_executor = nn_engine->CreateNNExecutor(model.model_file, model_list.at("minn_mnn").minn_name, model.key, config);   // 创建执行器对象
if (!nn_executor)
{
    AURA_LOGE(ctx, NN_TAG, "failed to create nn executor: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}

if (nn_executor->Initialize() != aura::Status::OK)                  // 初始化执行器对象
{
    AURA_LOGE(ctx, NN_TAG, "failed to initialize nn executor: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}
```

### 2.1.3 执行推理

执行推理之前，我们需要先准备输入和输出数据。在 AURA 中，输入数据和输出数据都是 Mat 类型，我们按 InceptionV3 的输入输出格式来创建 Mat。当 Mat 创建完成后，需要将实际输入数据加载到输入 Mat 中，调用`Load`函数来实现外部数据加载。
```cpp
aura::Mat src(ctx.get(), aura::ElemType::F32, {299,  299, 3});      // 创建输入数据对象，NN 推理时，数据格式按 NHWC 存储
aura::Mat dst(ctx.get(), aura::ElemType::F32, {  1, 1001, 1});      // 创建输出数据对象，NN 推理时，数据格式按 NHWC 存储
if (!src.IsValid() || !dst.IsValid())                               // 有效性检查
{
    AURA_LOGE(ctx, NN_TAG, "failed to create mat: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}

if (src.Load(model.input_file) != aura::Status::OK)                 // 从文件中读取输入数据
{
    AURA_LOGE(ctx, NN_TAG, "failed to load data from %s to mat\n", model.input_file.c_str());
    return aura::Status::ERROR;
}
```
在 NN 中，我们使用 MatMap 来封装输入和输出数据，MatMap 的键为模型节点名称，值则为上面创建的与之关联的 Mat。
```cpp
aura::MatMap input  = {{model.input_node[0],  &src}};               // 关联输入节点与输入数据
aura::MatMap output = {{model.output_node[0], &dst}};               // 关联输出节点与输出数据

if (nn_executor->Forward(input, output) != aura::Status::OK)        // 执行推理
{
    AURA_LOGE(ctx, NN_TAG, "failed to execute forward inference: %s\n", ctx->GetLogger()->GetErrorString().c_str());
    return aura::Status::ERROR;
}
```

### 2.1.4 推理验证

输入的示例数据是包含了类似垃圾箱物体的图片，通过查询我们可以知道垃圾箱的是词目是n02747177，对应的标签是413，我们据此编写`Validate`函数来验证推理结果。
```cpp
aura::Status Validate(const aura::Mat &mat)
{
    if (!mat.IsValid())                                             // 有效性检查
    {
        fprintf(stderr, "invalid mat\n");
        return aura::Status::ERROR;
    }

    MI_S32 id    = 0;                                               // 预测结果
    MI_F32 max_p = mat.Ptr<MI_F32>(0)[0];                           // 最大概率

    for (MI_S32 i = 1; i < mat.GetSizes().m_width; i++)
    {
        MI_F32 p = mat.Ptr<MI_F32>(0)[i];
        if (p > max_p)                                              // 找到概率最大的类别
        {
            max_p = p;
            id    = i;
        }
    }

    if (id != 413)                                                  // 验证预测结果，垃圾桶的索引为 413
    {
        fprintf(stderr, "the label should be 413(n02747177, trash bin), but it was given %d\n", id);
        return aura::Status::ERROR;
    }

    return aura::Status::OK;
}

if (Validate(dst) != aura::Status::OK)                              // 验证结果
{
    AURA_LOGE(ctx, NN_TAG, "failed to validate result\n");
    return aura::Status::ERROR;
}
```

## 2.2 使用CMake编译

上述示例的完整代码及相应的 CMakeLists.txt 文件可在 AURA 的安装目录下找到，开发者可以使用 CMake 来对示例代码进行编译。

由于 NN 需要依赖 AURA，因此在编译示例之前，需要先编译 AURA，我们定义 `Aura_DIR` 变量来表示 AURA 的安装目录。

示例演示 NN 在 Android 平台下的推理，在 CMakeLists.txt 中，我们首先对当前目标平台进行检测。
```cmake
cmake_minimum_required(VERSION 3.2)

if(${CMAKE_SYSTEM_NAME} NOT MATCHES "Android")
    message(FATAL_ERROR "the target system must be Android")
endif()

if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
```
接下来，我们需要调用 `find_package` 来查找 AURA 的依赖项，变量 `Aura_FOUND` 可以被作为查找成功的标志。查找成功后，我们对 `Aura_LIBS` 进行链接即可。
```cmake
project(aura_nn)

if(Aura_DIR)    # cmake path
    find_package(Aura REQUIRED)
    if(NOT Aura_FOUND)
        message(FATAL_ERROR "failed to find aura")
    endif()
else()
    message(FATAL_ERROR "aura path is empty")
endif()

add_executable(sample_nn ./main.cpp)

target_link_libraries(sample_nn ${Aura_LIBS})

install(TARGETS sample_nn DESTINATION sample_nn)
```
下面是一个可能的编译命令示例：
```
# Assuming the compilation parameters of aura are as follows
cmake                                                                                   \
    -DCMAKE_INSTALL_PREFIX=${Aura_DIR}                                                  \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake        \
    -DANDROID_ABI="arm64-v8a"                                                           \
    -DANDROID_PLATFORM=android-23                                                       \
    -DANDROID_STL=c++_shared                                                            \
    -DAURA_LIB_TYPE=share                                                               \
    -DAURA_ENABLE_NN=ON                                                                 \
    .. && make install

# Then the parameters for compiling NN can be as follows
cmake -DAura_DIR=${Aura_DIR}/cmake                                                      \
      -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake    \
      -DANDROID_ABI=arm64-v8a                                                           \
      -DANDROID_PLATFORM=android-23                                                     \
      .. && make
```

## 2.3 运行示例

在编译完成后，我们使用adb分别将模型、测试数据、库文件(仅动态链接需要)以及可执行文件推送到目标设备中，本示例的目标路径为`/data/local/tmp/aura`。
```
adb shell mkdir -p /data/local/tmp/aura

adb push data              /data/local/tmp/aura
adb push libaura.so        /data/local/tmp/aura # 仅动态链接需要
adb push libmnn_wrapper.so /data/local/tmp/aura # 仅 MNN 框架需要
adb push sample_nn         /data/local/tmp/aura
```

执行下面命令，即可运行示例
```
adb shell "cd /data/local/tmp/aura; export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH; chmod +x sample_nn; ./sample_nn minn_mnn"
```
为了方便用户使用，我们提供了`build_and_run.sh`脚本，在nn的安装目录使用命令行运行该脚本将自动完成上述步骤。

# 3. 常见问题

- 如何将MNN/NP/QNN/SNPE/XNN模型转换为NN模型：使用[模型转换工具](https://xiaomi.f.mioffice.cn/docx/doxk4ayvzND33eJvTFQMrY4mqFb)
