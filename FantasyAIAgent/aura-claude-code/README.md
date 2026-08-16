# aura-claude-code

面向 Claude/Codex 类编码代理的项目级工具集合。目前包含一个 native crash 分析 Skill：

```text
project-skills/crash-analysis/SKILL.md
```

该 Skill 约定从目标工程的 `logs/crash/` 读取 crash 日志和带符号 `.so`，使用 BuildId、
`readelf`、`addr2line` 等信息生成中文分析报告。它是模板资源，不会自动安装到全局环境。

使用前请检查 Skill 中的默认目录是否符合目标工程，并确保 crash 日志不包含需要脱敏的用户
数据或密钥。新增 Skill 时放在 `project-skills/<skill-name>/SKILL.md`，同时在本 README
登记用途和必要输入。
