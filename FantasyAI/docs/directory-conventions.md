# FantasyAI 目录约定

## 1. 推荐结构

新建子工程使用以下结构；只创建确实有内容的目录，不提交空目录：

```text
ProjectName/
├── README.md              # 定位、快速开始、依赖、平台边界
├── docs/                  # 扩展文档，非入口 README
├── src/<package>/         # 可复用 Python/C++ 代码
├── examples/              # 可直接执行的 demo / Notebook
├── tests/                 # 有断言、无隐式外部副作用的测试
├── configs/               # YAML/INI/DATA 等非敏感配置
├── scripts/               # 下载、转换、部署等辅助命令
├── assets/                # README/demo 所需的小型图片和固定样例
├── requirements.txt       # 运行依赖（或 pyproject.toml）
└── requirements-dev.txt   # 测试/质量工具依赖
```

## 2. 文件归类

| 内容 | 应放位置 | 不应放位置 |
|---|---|---|
| 快速入门 | 子工程 `README.md` | 只放在仓库顶层 |
| 原理/调研文章 | `docs/*.md` | `read.md`、项目根目录随意命名 |
| Notebook | `examples/` 或 `notebooks/` | 与包源码混在同一层 |
| 单元测试 | `tests/test_*.py` | 用 `test.py` 表示训练评估入口 |
| 评估/性能脚本 | `scripts/evaluate_*.py` 或清晰的 CLI 入口 | 被 pytest 误收集 |
| 训练产物 | `outputs/`, `runs/`, `checkpoints/` | Git 跟踪的源码目录 |
| 模型/数据集 | 外部存储，README 记录获取方式与校验值 | Git 仓库 |
| 密钥/私有路径 | 环境变量、本地配置 | 源码和公共 INI/YAML |

## 3. Python 约定

- 可复用代码放在 `src/`，为包目录添加 `__init__.py`。
- 包内导入使用相对导入；命令行入口放在包外，避免修改 `sys.path`。
- 输入、模型和输出路径使用 `argparse` + `pathlib.Path`，不写 `/home/...`、`/Users/...` 或盘符绝对路径。
- 脚本导入时不应立即训练、下载、打开 GUI 或连接设备；副作用放入 `main()`。
- 真正的单测不需模型下载、公网、摄像头和 GPU；环境测试单独标记。

## 4. C++ / CMake 约定

- 使用 out-of-source build：`cmake -S . -B build`。
- 第三方库使用 `find_package`/`find_library`，非标准位置通过 CMake cache 参数或环境变量传入。
- 不在 `CMakeLists.txt` 中保存个人电脑的 OpenCV/ONNX Runtime 绝对路径。
- 每个可执行 demo 建立独立 target，共享逻辑抽成 library target。

## 5. 兼容历史代码的迁移方式

1. 先补 README/文档索引和 `.gitignore`，确立新文件规则。
2. 再处理无导入关系的文档、Notebook 和辅助脚本。
3. 对训练/推理核心代码，先添加 smoke test，再移入 `src/`，并保留一个参数兼容入口。
4. YOLO 等上游整包优先保留原布局，本地代码放到外层 `scripts/`/`docs/`，避免上游升级困难。

## 6. 自动检查

```bash
cd FantasyAI
python tools/check_structure.py
```

检查内容包括：每个子工程必须有 README、非 README Markdown 必须在 `docs/`、所有 Python 文件可被当前解释器解析，以及 Notebook 是合法的 JSON/nbformat 结构。
