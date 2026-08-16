# FantasyShell

Android/QNX/SoC 开发的个人命令工具箱。多数脚本会调用 `adb`、`ssh`、`scp`、芯片 SDK 或设备上的命令，因此“Shell 语法通过”不代表在没有目标设备时能完成业务操作。

## 命令分类

| 类别 | 代表脚本 | 外部依赖 |
|---|---|---|
| ADB 会话 | `adb-tcp`, `ashell`, `logcat`, `kcam` | adb + Android 设备 |
| 日志与 crash | `asdlog`, `osdlog`, `logsd`, `grep-asd-crash` | adb/设备日志目录 |
| CDSP/QNN/SNPE | `cdsp`, `restart_cdsp`, `qcomboost`, `snpe-verbose` | Qualcomm 目标机/SDK |
| MTK/镜像 | `mtkboost`, `checkrom` | MediaTek 或 Android 工具 |
| 文件与哈希 | `file_all`, `md5`, `md5_all`, `check_strip` | GNU/BSD 命令差异需注意 |
| SSH/QNX | `ssh-ai`, `mini-dm*` | ssh/scp/目标机 |
| 地址解析 | `addr*`, `addr_parser*` | addr2line/Python，符号文件 |

## 使用

```bash
# 先查看内容和参数，再单独调用
cd FantasyShell
bash scripts/addr.sh <args>

# 仅检查某个脚本的语法，不连接设备
bash -n scripts/addr.sh
```

macOS 使用 Bash 3.2/BSD 命令，Linux 通常使用新版 Bash/GNU coreutils；脚本不应假定 `md5sum` 或 `md5` 在两端同时存在。Windows 需 Git Bash 或 WSL，原生 `cmd.exe`/PowerShell 不能直接执行这些 Bash 脚本。

部分历史文件没有 shebang，或只是用于 `source` 的函数/别名片段。请勿将整个 `scripts/`
目录无差别加入 PATH；更完整的环境分层见 `FantasyToolkits/aura-dev-env`。
