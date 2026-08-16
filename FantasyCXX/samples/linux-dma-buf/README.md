# Linux DMA-BUF exporter 学习草稿

`exporter_dummy.c` 演示 Linux 内核 DMA-BUF exporter 的回调结构。当前回调大多是占位实现，
没有完整的资源分配与释放逻辑，不能作为生产内核模块加载。

该示例依赖 Linux 内核头文件与目标内核版本对应的 Kbuild 环境，macOS 和 Windows 不支持。
它没有加入 `FantasyCXX` 的默认 CMake 构建，避免在非 Linux 平台破坏现有工程构建。
