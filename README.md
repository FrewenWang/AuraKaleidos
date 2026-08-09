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

聚合仓库根目录不提供统一编译入口。
