# AuraKaleidos

这是一个代码聚合仓库。根目录下的各一级目录是彼此独立的工程或同类示例集合，不存在统一的依赖、
构建顺序或发布关系。

请进入具体子工程，根据该工程的 `README.md` 和语言工具链单独安装依赖、构建与测试。例如：

```bash
cd FantasyAlgorithm/CXX
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

聚合仓库根目录不提供统一编译入口，但提供无外部副作用的结构与仓库卫生检查：

```bash
python tools/repository_check.py
python -m unittest discover -s tests -v
```

## 一级工程

- `AliceAndroid`、`AliceJava`、`FantasyKotlin`、`FantasyFlutter`、`FantasySwiftIOS`：客户端与语言示例。
- `FantasyAI`、`FantasyHPC`、`FantasyCuda`、`FantasyAutoDrive`：AI、异构计算与自动驾驶工程。
- `FantasyAlgorithm`、`FantasyCXX`、`FantasyPython`、`FantasyJS`、`FantasyNodeJS`：算法与语言工程。
- `AliceAutoTest`：Phoenix 自动化测试工程，完整业务流程主要面向 Windows，macOS/Linux 可运行离线单元测试。
- `FantasyAIAgent`：独立的 AI Agent 工具与技能集合。
- `FantasyShell`、`FantasyToolkits`：Shell 命令与开发环境工具。

大模型、数据集、测试视频和平台工具二进制不纳入 Git，也不使用 Git LFS；请根据各子工程说明单独准备。

完整的分层测试矩阵、平台工具链边界、CI 覆盖和提交规范见
[工程维护与测试指南](docs/工程维护与测试指南.md)。

## AliceKaleidos 合并记录

`/Users/frewen/01.WorkSpace/AliceKaleidos` 已按子工程职责合并到本仓库；目录映射、取舍原则、
平台边界和验证方式见 [工程合并说明](docs/工程合并说明.md)。
