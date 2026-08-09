# AuraKaleidos 工程审计与优化记录

## 覆盖范围

本次以 `config/projects.json` 中的第一方工程为可执行清单，共覆盖 C/C++、CUDA、Python、AI/HPC、
JavaScript/TypeScript、Java、Android/Kotlin、Flutter、Swift 和 Shell/设备工具。第三方子模块、上游
Jetson/YOLO 镜像及 Xcode 工程内部结构不做机械迁移，以免破坏其工具链。

## 已实施的统一规则

| 工程类型 | 正规目录 | 本次落实 |
| --- | --- | --- |
| C/C++/CUDA | `include/`, `src/`, `tests/`, `examples/`, `assets/` | 算法题与 OpenCV 演示移入 `examples/`；CUDA 在无 Toolkit 平台可正常跳过 |
| Python | `src/<package>/`, `tests/`, `examples/`, `pyproject.toml` | 五个小型包补齐打包元数据；目标跟踪资源移入包内 `assets/` |
| Node.js | `src/`, `tests/`, `public/`, `views/`, `outputs/` | 服务入口与应用对象分离；爬虫去除废弃请求库并固定下载目录 |
| Java/Android/Kotlin | Maven 或 Gradle 官方结构 | AliceJava 补齐可构建骨架与 JUnit；Android 工程保留标准 module 布局 |
| Flutter/Swift | 工具链官方结构 | 保留 `lib/test`、Xcode target 与资源目录，避免无收益改名 |
| Shell/工具 | `scripts/` 或 CLI 的 `bin/` | Shell 增加统一语法测试；设备 CLI 保留可直接执行入口 |
| AI/HPC | `src/`, `scripts/`, `models/`, `data/`, `logs/`, `outputs/` | 模型与数据目录职责明确；NVIDIA/Jetson 工程标注平台限制 |

## 平台策略

- macOS：运行通用 C++、Python、Node、Java 语法/单测；OpenCV 使用仓库中匹配架构的预编译库。
- Linux/NVIDIA：额外启用 CUDA、TensorRT、Jetson 和设备部署工程。
- Android：由 Gradle/Android SDK 构建；iOS 由 Xcode 构建。
- CMake 工程继续复用 `FantasyCXX/cmake/AuraPlatform.cmake` 的系统与架构参数。

## 仍需治理的数据问题

仓库历史中仍存在已被 `.gitignore` 覆盖、但此前已经提交的大型 ONNX、模型缓存、数据集和平台二进制。
目录检查器会报告它们。`.gitignore` 只阻止新文件，不能自动取消已经跟踪的文件；后续应使用
`git rm --cached <明确文件>` 分批取消跟踪，文件会保留在本地，且不启用 Git LFS。

## 自动检查

```bash
python3 scripts/check_project_layout.py
bash scripts/test_macos.sh
```

普通检查把历史大文件作为警告；CI 可使用 `--strict`，待历史文件全部取消跟踪后再启用。
