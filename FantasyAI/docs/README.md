# FantasyAI 文档

本目录保存 FantasyAI 工程级文档。每个子工程的使用入口仍保留在子工程根目录 `README.md`。

- [目录约定](directory-conventions.md)：新增或整理代码时应遵守的结构。
- [子工程盘点](subprojects.md)：11 个子工程的责任、入口、当前布局与例外。

文档放置原则：

- `README.md`：项目/独立组件入口，与被说明的代码同级。
- `docs/`：架构、算法原理、数据格式、调试记录和较长的专题文章。
- 代码内 docstring：函数/类的 API 约束，不用 Markdown 替代。
