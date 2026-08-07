# AuraKaleidos 工程目录规范

本仓库是多语言、多个独立工程组成的 monorepo。目录整理遵循各语言生态的默认约定，
不要求每个工程机械地创建所有目录；只有存在对应内容时才创建目录。

## 通用职责

| 目录 | 用途 | 是否提交到 Git |
| --- | --- | --- |
| `src/` | 可复用源码、业务实现 | 是 |
| `include/` | C/C++ 对外头文件 | 是 |
| `examples/` | 可直接运行的教学或示例代码 | 是 |
| `scripts/` | 构建、训练、转换、部署和运维脚本 | 是 |
| `tests/` | 单元测试和集成测试 | 是 |
| `notebooks/` | Jupyter notebook | 是 |
| `assets/` | 图片、配置模板等静态资源 | 小文件提交 |
| `data/` | 输入数据、测试夹具 | 仅提交小型夹具和说明文件 |
| `models/` | 模型文件 | 默认忽略，不使用 Git LFS |
| `logs/` | 运行日志 | 忽略，仅保留 `.gitkeep` |
| `outputs/` | 报告、图片、下载和推理结果 | 忽略，仅保留 `.gitkeep` |
| `build/`、`dist/` | 编译和打包产物 | 忽略 |

## 各语言工程约定

### C、C++ 和 CUDA

采用 `include/ + src/ + tests/ + examples/ + scripts/ + assets/`。CMake 入口保留在工程根目录，
顶层 `build.sh` 可以作为公开构建入口保留；模块内部紧邻实现的 `test/` 目录不强制迁移。

- `FantasyCXX`：自研顶层测试已统一为 `tests/`，示例与脚本采用复数目录；`aura-cv`、
  `aura-vision` 等设备 SDK 保持其现有分层，避免破坏平台构建脚本。
- `FantasyCXX/3rdparty`：完整排除在目录重排之外。第三方源码必须保持上游布局。
- `FantasyCuda`：CUDA 源码位于 `src/`，notebook 位于 `notebooks/`；本地 ELF 可执行文件不再跟踪。
- `FantasyAlgorithm/CXX`：算法源码位于 `src/`，历史测试归入 `tests/`。

### Python、AI 和 HPC

可复用模块进入 `src/`，一次性任务进入 `scripts/`，教学代码进入 `examples/`，notebook 进入
`notebooks/`。模型、数据、日志和结果分别进入 `models/`、`data/`、`logs/` 和 `outputs/`。

- `FantasyAlgorithm/python`：`src/ + tests/`。
- `FantasyAutoDrive`：插值、协方差等教学代码进入 `examples/`；卡尔曼滤波项目使用
  `src/ + examples/ + assets/`。
- `kalman_filter_with_yolo11_objects_tracker`：使用
  `src/ + tests/ + data/videos + models + outputs/figures + docs/`；入口 `track.py` 留在根目录。
- `FantasyAI/AliceAILearn`：课程主体是 notebook，因此使用 `notebooks/ + assets/ + dataset/ + logs/`。
  `dataset/` 是机器学习领域常用名称，可与 `data/` 等价使用。
- `FantasyAI/AlicePyTorch`：根目录课程代码进入 `examples/`，notebook 进入 `notebooks/`，
  图片进入 `assets/images/`；已有章节式教程保持原结构。
- `FantasyAI/AlicePaddlePaddle`、`AliceTensorFlow`、`AliceTensorflow2.0`：示例和 notebook 已分离，
  TensorFlow 2.0 的课程 notebook 位于 `notebooks/examples/`；
  已按专题组织的子目录保持不变。
- `FantasyAI/AliceBaiduFaceDetection`、`AliceKaleidoYOLO`：属于完整训练代码或上游工程，模块间存在大量
  相对导入，保持包结构，只对新增数据、模型、日志和输出应用通用约定。
- `FantasyHPC`：演示进入 `examples/`；蒸馏工程使用 `src/ + scripts/ + data/`；模型转换脚本进入
  `scripts/`，ONNX 文件进入 `models/`。
- `FantasyHPC/alice-jetson-inference`：上游 CMake 工程，保持其 `c/ + python/ + tools/ + examples/` 布局。
- `FantasyPython/alice-pyopencv`：使用
  `src/ + examples/ + notebooks/ + assets/images + data/test_data`。
- `FantasyPython/aura-data-compare`：使用 `src/ + scripts/ + tests/ + data/`。
- `FantasyPython/aura-qdb`、`aura-midb`、`aura-toolkit`：命令入口 `bin/` 和安装脚本 `scripts/`
  已符合工具型 Python 工程惯例，环境初始化脚本保留在根目录以便 `source`。

### JavaScript、TypeScript 和 Node.js

业务源码进入 `src/`，静态站点资源保留在 `public/`，服务端模板保留在 `views/`，测试使用 `tests/`，
下载和生成结果进入 `outputs/`。`package.json` 始终保留在工程根目录。

- `FantasyJS`：库源码位于 `src/`，浏览器示例和静态资源保留在根目录与 `public/`。
- `AuraNodeCli`、`MySpiderDemo`、`MyPicSpider`：入口位于 `src/`；CLI 的 `bin` 映射已同步，爬虫下载进入
  `outputs/downloads/`。
- `MyMovieWeb`：Express 源码位于 `src/`，`public/` 和 `views/` 留在根目录；启动命令已同步。
- `NyxTSExpress`：原本已有标准 `src/`，测试统一为 `tests/` 并同步 Jest 配置。
- `myExpressGenarator`：保持 Express Generator 的官方 `app.js + bin/ + routes/ + public/ + views/` 布局。
- `NodeJsSamples`：它本身是示例集合，各示例子目录不再额外嵌套一层 `examples/`。

### Android、Kotlin、Flutter、Java 和 iOS

这些生态已有强约束目录，优先遵循工具链，不套用通用 `src/` 规则。

- `AliceAndroid`、`FantasyKotlin`：保留 Gradle 的 `app/src/main`、`app/src/test`、
  `app/src/androidTest` 和多 module 布局。
- `FantasyFlutter`：保留 Flutter 的 `lib/ + test/ + android/ + ios/ + assets/`；Flutter 官方目录名是
  单数 `test/`，不改为 `tests/`。
- `AliceJava`：Java 源码应按构建方式采用 Maven/Gradle 的 `src/main/java + src/test/java`；纯教学示例可放
  `examples/`。
- `FantasySwiftIOS`：保留 Xcode 工程组、`Assets.xcassets`、`*Tests` 和 `*UITests`，文件移动必须同时修改
  `project.pbxproj`，因此不做无收益重排。

### Shell 和工具集

- `FantasyShell`：所有独立 shell 工具进入 `scripts/`。
- `FantasyToolkits`：QDB/ODB 等工具继续使用 `bin/ + scripts/`；平台二进制和本地安装产物不提交。

## 新增工程检查清单

1. 先使用语言工具链的标准布局，再应用本文件的通用目录。
2. 不提交模型、数据集、日志、下载、编译产物、IDE 缓存和凭证。
3. 小型测试夹具可以提交到 `tests/fixtures/` 或 `data/fixtures/`。
4. 所有移动都要同步 CMake、Gradle、Xcode、`package.json`、导入路径和文档链接。
5. 脚本和程序中的资源路径应基于脚本文件或工程根目录计算，不依赖调用者的当前目录。
6. 上游镜像、submodule 和 vendored 代码不做本地目录重排。
