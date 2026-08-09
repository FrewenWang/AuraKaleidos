# FantasyShell

可独立执行的 Shell 工具位于 `scripts/`。脚本应使用 `set -euo pipefail`，通过脚本自身位置解析资源，
并通过 `bash -n` 静态语法检查。

```bash
bash tests/test_scripts.sh
```
