# FantasyAIAgent 开发说明

本目录聚合 AI 编码代理相关的项目级工具与 Skills，不是可统一构建的软件包。

## 目录约定

- 每个工具使用独立子目录，例如 `aura-claude-code/`。
- 可复用 Skill 放在 `<tool>/project-skills/<skill-name>/SKILL.md`。
- 工具入口、安装方式、触发条件和必要输入写在对应子目录 README。
- 运行产生的日志、crash、带符号 SO、分析报告和用户数据不提交 Git。

## 编辑要求

- Skill 的 YAML frontmatter 必须包含稳定的 `name` 和清晰的 `description`。
- 指令必须区分事实、推断和待确认项，不允许在缺少输入时编造结论。
- 示例路径使用相对路径或占位符，不写作者机器的绝对路径。
- 涉及日志、截图、凭据或业务数据时，先说明脱敏要求。

当前主要工具见 [`aura-claude-code/README.md`](aura-claude-code/README.md)。
