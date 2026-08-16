# AURA Dev Environment

多环境开发工具，当前管理 `dev`、`mi`、`qnx` 三个已注册环境，并在所有环境中加载 `base` 配置。

## 🚀 快速开始

### 安装（一条命令）
```bash
./setup.sh
```

### 使用

**打开新的 terminal 或重新加载：**
```bash
source ~/.zshrc              # macOS 默认 zsh
# source ~/.bashrc           # Linux bash
```

**常用命令：**
```bash
aura-env list              # 列出所有环境
aura-env current           # 查看当前环境
aura-env select dev        # 切换到 dev 环境
aura-env show              # 查看当前配置
aura-env validate          # 验证环境
aura-env help              # 显示帮助
```

## 📁 项目结构

```
aura-dev-env/
├── setup.sh                          # 安装脚本
├── aura-dev-env.sh                   # 核心环境加载脚本
├── scripts/
│   ├── env-manager.sh               # 环境管理工具
│   ├── init-shell.sh                # 安全初始化
│   └── repair-shell-init.sh         # 诊断修复工具
├── config/
│   ├── environments.conf            # 已注册环境列表
│   └── toolkit_versions.conf        # 工具版本记录
├── bash/
│   ├── base/                        # 基础 Shell 配置
│   ├── dev/                         # 通用开发配置
│   └── mi/                          # Mobile Infrastructure 配置
└── bin/
    ├── base/                        # 共用工具
    ├── dev/                         # 开发环境工具
    ├── mi/                          # Mobile Infrastructure 工具
    └── qnx/                         # QNX 设备工具
```

## 📖 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/快速开始.md) | ⭐ 最简化的使用指南（5分钟） |
| [shell初始化参考](docs/shell初始化快速参考.md) | Shell 初始化参考 |
| [命令速查](docs/速查表.md) | 常用命令速查 |
| [命令参考](docs/命令参考.md) | 命令详细说明 |
| [架构设计](docs/架构设计.md) | 加载顺序和路径分层 |

## 🔧 故障排查

**问题：aura-env 命令找不到**
```bash
# 方式1: 打开新的 terminal
# 方式2: 重新加载
source ~/.zshrc  # macOS；Linux bash 使用 source ~/.bashrc
```

**问题：初始化报错**
```bash
bash scripts/repair-shell-init.sh .
```

**问题：环境配置不正确**
```bash
aura-env validate
```

## 📋 支持的环境

| 环境 | 说明 | 用途 |
|------|------|------|
| base | 基础层 | 所有环境自动加载，不在注册列表中 |
| dev | 开发环境 | 本地开发、调试、测试 |
| mi | Mobile Infrastructure | Android/SoC 开发工具 |
| qnx | QNX 工具 | 只有 `bin/qnx`，无额外 Shell 配置 |

QNX 登录脚本不保存密码和设备地址。使用前设置 `AURA_QNX_HOST`，并通过 `SSHPASS` 临时
传入密码；条件允许时优先配置 SSH Key。

## 🔧 配置环境列表

环境列表在 `config/environments.conf` 中定义，可以通过编辑该文件来添加或移除环境：

```bash
# config/environments.conf
dev
mi
qnx
```

**添加新环境的步骤：**
1. 在 `config/environments.conf` 中添加环境名称
2. 按需创建 `bash/{new-env}/` 和 `bin/{new-env}/` 目录
3. 在 `bash/{new-env}/` 中创建 `bashrc.fwf` 或平台专用配置
4. 将命令脚本放入 `bin/{new-env}/`
5. 运行 `aura-env list` 验证新环境出现

## ⚙️ 配置

环境配置存储在：
- `~/.aura-env-config` - 当前选择的环境
- `config/environments.conf` - 环境注册列表
- `bash/<environment>/` - 环境 Shell 配置

切换环境后需要重新加载：
```bash
aura-env select <environment>
source ~/.zshrc  # 或 Linux 上的 ~/.bashrc
```

## 🆘 获取帮助

```bash
# 显示所有命令
aura-env help

# 查看快速参考
cat docs/快速开始.md

# 完整文档
cat docs/shell初始化快速参考.md
```

## 📝 许可证

MIT License

---

**提示：** 首次使用，请阅读 [快速开始](docs/快速开始.md)。Windows 用户需使用 WSL/Git Bash；本项目不修改 PowerShell profile。
