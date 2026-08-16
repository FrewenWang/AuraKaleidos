# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在此仓库中工作时提供指引。

## 项目概述

AURA 是一个多环境开发框架，管理不同的开发环境（dev、mi 等），支持共享基础配置和环境/平台特定的覆盖。它提供 shell 别名、环境变量和开发工具，具有智能分层和优雅降级。

**核心原理**：配置通过 4 层继承（基础 → 平台 → 环境 → 环境+平台），后面的层覆盖前面的层。缺失的目录/文件不会破坏系统。

## 架构：4层配置系统

系统使用两个平行的 4 层结构：

### Bash 配置加载（aura-dev-env.sh，第 66-100 行）
1. **第1层**：`bash/base/bashrc.fwf` - 通用别名、变量、git 快捷键
2. **第2层**：`bash/base/bashrc_{Platform}.fwf` - 平台特定（如 Linux `open` 命令）
3. **第3层**：`bash/{env}/bashrc.fwf` - 环境特定配置（dev/mi 等）
4. **第4层**：`bash/{env}/bashrc_{Platform}.fwf` - 环境 + 平台组合

后面的层覆盖前面的层（标准 bash 优先级）。缺失的文件通过 `[ -f ... ] && source` 模式被静默跳过。

### 二进制/脚本 PATH 构建（aura-dev-env.sh，第 106-141 行，build_bin_path 函数）
1. `bin/{env}/{Platform}/` - 最高优先级（环境 + 平台工具）
2. `bin/{env}/` - 环境特定工具
3. `bin/base/{Platform}/` - 平台特定基础工具
4. `bin/base/` - 通用工具
5. `bin/main/` - 历史兼容性
6. 系统 PATH - 系统二进制文件

`build_bin_path()` 函数仅添加存在的目录，防止 PATH 污染。在保留顺序的同时删除重复项。

## 核心命令

```bash
# 安装与设置
bash setup.sh                    # 一次性安装，自动检测 OS/shell，更新 ~/.bashrc

# 环境管理
aura-env list                    # 显示所有可用环境
aura-env current                 # 显示当前激活的环境
aura-env select <env>            # 切换环境（例如 dev、mi）
aura-env show [env]              # 显示环境配置文件（✓=存在，○=缺失）
aura-env validate [env]          # 验证 bash 语法，优雅处理缺失文件
aura-env info [env]              # 显示路径和文件状态

# 切换环境后
source ~/.bashrc                 # 重新加载以应用更改

# 诊断
bash scripts/repair-shell-init.sh .    # 完整的环境检查和诊断
```

## 核心组件

### 主要脚本
- **setup.sh**：初始安装。自动检测 OS（Linux/Darwin）和 shell（bash/zsh），用 AURA_ENV 块更新 rc 文件，幂等。
- **aura-dev-env.sh**：由 rc 文件 source。加载 4 层 bash 配置，构建 4 层 PATH，导出 AURA_ENV_ROOT，创建 `aura-env` 命令别名。
- **scripts/env-manager.sh**：环境操作的 CLI 工具。从 `config/environments.conf` 读取已注册的环境。
- **scripts/init-shell.sh**：带验证和错误处理的安全初始化。
- **scripts/repair-shell-init.sh**：诊断工具，检查文件存在性和语法。

### 配置文件
- **config/environments.conf**：可用环境的权威列表（每行一个）。由 setup.sh 和 env-manager.sh 使用。
- **~/.aura-env-config**：用户当前的环境选择，由 `aura-env select` 创建。

### Bash 配置层（bash/）
- **bash/base/bashrc.fwf**：60+ 个通用别名（ll、la、cd..、grep、git 快捷键）、基础变量
- **bash/base/bashrc_{Linux|Darwin}.fwf**：平台特定（例如 `alias open="nautilus"`）
- **bash/{dev,mi}/bashrc.fwf**：环境特定的别名、变量、编译标志
- **bash/{dev,mi}/bashrc_{Linux|Darwin}.fwf**：该环境的 Linux/macOS 特定工具

### 二进制/脚本层（bin/）
- **bin/base/**：60+ 个通用工具、版本管理
- **bin/base/{Linux|Darwin}/**：adb、fastboot、git、ndk、scrcpy 等
- **bin/{dev,mi}/**：debug-adb、mi-build、mi-deploy
- **bin/{dev,mi}/{Linux|Darwin}/**：平台特定工具（debug-build、perf-profile、adb-manager 等）

## 开发任务

### 添加新环境
1. 将环境名称添加到 `config/environments.conf`
2. 创建 `bash/{env_name}/{Linux,Darwin}/` 目录
3. 创建配置文件：
   - `bash/{env_name}/bashrc.fwf`（环境特定）
   - `bash/{env_name}/bashrc_Linux.fwf` 和 `bashrc_Darwin.fwf`（可选）
4. 创建 `bin/{env_name}/{Linux,Darwin}/` 并添加脚本（可选）
5. 测试：`aura-env list` 应显示新环境

### 为环境添加工具
1. 将脚本添加到 `bin/{env}/{platform}/`（例如 `bin/dev/Linux/my-tool`）
2. 使其可执行：`chmod +x bin/dev/Linux/my-tool`
3. 重新加载 shell：`source ~/.bashrc`

### 为环境添加别名或变量
1. 编辑 `bash/{env}/bashrc.fwf` 或 `bash/{env}/bashrc_{Platform}.fwf`
2. 重新加载：`source ~/.bashrc`

### 调试环境问题
```bash
# 检查加载的内容
aura-env show dev

# 验证语法
bash -n bash/dev/bashrc.fwf
bash -n bash/dev/bashrc_Linux.fwf

# 完整诊断
bash scripts/repair-shell-init.sh .

# 检查 PATH
echo $PATH | tr ':' '\n' | grep aura-dev-env
```

## 重要设计模式

### 优雅降级
- 缺失的环境目录不会破坏系统；它们被跳过
- 如果 `bash/{env}/bashrc.fwf` 不存在，配置回退到基础
- 如果 `bin/{env}/{platform}/` 不存在，PATH 仅包含现有目录
- 环境可以在 config/environments.conf 中注册，然后再创建目录

### 文件存在检查
- 所有 sourcing 使用 `[ -f "$file" ] && source "$file"` 模式
- 所有目录添加使用 `[ -d "$dir" ] && new_path="$new_path:$dir"` 模式
- 缺少文件时从不失败；总是 `|| true` 回退

### 平台检测
- `uname -s` 返回"Linux"或"Darwin"（macOS）
- 用于选择平台特定文件：`bashrc_${PLATFORM}.fwf`

### 幂等性
- setup.sh 检查 rc 文件中是否已有 AURA_ENV 块；如果存在则跳过
- aura-env select 覆盖 ~/.aura-env-config；总是一致
- 多次 source aura-dev-env.sh 是安全的（导出是幂等的）

## 文件命名约定

- `.fwf` 扩展名："Frewen.Wang 的自定义文件"（在所有 bash 配置中使用）
- 命名：英文用于脚本/工具，文档优先使用中文

## 测试框架

```bash
# 测试新环境
aura-env select <new-env>
source ~/.bashrc
aura-env current
aura-env show <new-env>
aura-env validate <new-env>

# 验证 PATH
which <tool-name>
$(<tool-name>)

# 检查别名
alias
ll  # 如果加载，应该有效
```

## 文档

- **docs/快速开始.md**：5 分钟快速开始、常见任务
- **docs/架构设计.md**：深入的 4 层架构、优先级规则、添加环境
- **docs/命令参考.md**：命令参考、故障排除指南
- **docs/速查表.md**：所有工具和别名的快速查询
- **docs/shell初始化快速参考.md**：Shell 特定的初始化参考

## 最近的改进（最新提交）

1. **4 层架构**（e30412b）：实现了完整的 4 层 bash 和 bin 层级，支持环境/平台覆盖
2. **优雅降级**（dcbcaea）：缺失的目录不会破坏系统；环境可以部分工作或使用继承的配置
3. **项目清理**（fc26a8b）：删除了冗余脚本，将文档从 10 个合并到 5 个，减少了 67% 的文档
4. **中文文档**（627397c）：所有文档转换为中文以适应用户群体

## 常见陷阱要避免

1. **不要强制目录存在**：系统优雅地处理缺失的目录；不要添加不必要的检查
2. **不要硬编码路径**：对相对路径使用 `$(dirname "$SCRIPT_SOURCE")`，绝不使用绝对路径
3. **不要忘记平台变体**：添加工具时，考虑 Linux 和 Darwin 版本
4. **不要重复基础配置**：环境继承基础别名/变量；仅覆盖不同的部分
5. **不要在脚本内重新加载 shell**：让用户决定何时重新加载；提供清晰的说明

## 其他注意事项

1. **请使用中文进行解答**：