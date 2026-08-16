# FantasyAIAgent

AI 编码代理相关工具、项目级 Skills 和使用笔记的聚合子工程。每个代理工具在自己的目录中
维护说明、Skill 定义和必要资源；本目录没有统一构建入口。

## 目录

```text
FantasyAIAgent/
├── README.md
├── CLAUDE.md
└── aura-claude-code/
    ├── README.md
    └── project-skills/
        └── crash-analysis/
            └── SKILL.md
```

## 当前内容

- [`aura-claude-code/`](aura-claude-code/)：Claude/Codex 类编码代理的项目级工具。
- [`crash-analysis`](aura-claude-code/project-skills/crash-analysis/SKILL.md)：结合 crash 日志、
  带符号 SO 和工程源码进行 Android/native crash 分析。

## CLI 命令笔记

以下命令仅作为历史速查，具体是否可用取决于所使用的 CLI 或桌面客户端：

| 命令 | 常见用途 |
|---|---|
| `/clear` | 清理当前会话上下文 |
| `/cost` | 查看成本或用量 |
| `/login`, `/logout` | 登录与退出 |
| `/model` | 查看或选择模型 |
| `/status` | 查看当前状态 |
| `/doctor` | 运行环境诊断 |

这些命令不是本工程实现的程序。新增代理工具时应创建独立子目录，并在其 README 中写明
安装位置、触发条件、输入、输出和安全边界。
