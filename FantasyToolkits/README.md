# FantasyToolkits

开发环境和设备工具的聚合工程。核心子工程 `aura-dev-env/` 根据选中环境和宿主系统组合 Shell 配置与 PATH。

## 加载逻辑

```text
setup.sh
  └── 写入 ~/.bashrc | ~/.bash_profile | ~/.zshrc
        └── source aura-dev-env.sh
              ├── bash/base/bashrc.fwf
              ├── bash/base/bashrc_<Darwin|Linux>.fwf
              ├── bash/<selected>/bashrc*.fwf
              └── bin/<selected>/<platform> + bin/base + PATH
```

`scripts/env-manager.sh` 读写 `~/.aura-env-config`，目前注册环境为 `dev`、`mi`、`qnx`。`qnx` 只有工具目录，没有额外 Shell 配置。

## 安装和验证

```bash
cd FantasyToolkits/aura-dev-env
bash setup.sh dev
source ~/.zshrc                       # macOS zsh
# source ~/.bashrc                    # Linux bash

aura-env list
aura-env select mi
aura-env validate
```

安装脚本会修改 Shell rc 文件，首次运行会对旧文件做时间戳备份。测试和验证应在
`FantasyToolkits/aura-dev-env` 子工程内完成；需要隔离验证时，请显式使用临时 HOME，避免
修改真实用户配置。

## 平台与风险

- 环境加载器支持 macOS/Linux 的 Bash/Zsh。Windows 需通过 WSL 或 Git Bash 使用，不是原生 PowerShell 配置器。
- `bin/` 中既有脚本也可能有平台二进制。随仓不代表它与当前 CPU/系统版本兼容，使用前运行 `<tool> --version`。
- 将 `bin/base` 放在 PATH 前部会覆盖同名系统命令。遇到 `md5`、`log`等通用名称时用 `type -a <name>` 确认实际解析结果。
- 工具在 `FantasyPython/aura-midb`/`aura-qdb` 中存在历史副本；新增工具应优先集中在本工程。

详细命令和内部结构见 [`aura-dev-env/README.md`](aura-dev-env/README.md) 与其 `docs/`。
