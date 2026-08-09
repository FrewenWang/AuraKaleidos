# aura-vision

跨平台视觉推理工程。核心实现位于 `src/`，示例与工具位于 `samples/`、`demo/` 和 `tools/`，
平台适配集中在 `platforms/` 与 Android module 中。构建参数由顶层 CMake 和 `AuraPlatform.cmake` 控制。

模型、校准缓存、构建结果和平台工具二进制均由 Git 忽略。
