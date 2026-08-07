# AuraNodeCli

命令行入口位于 `src/index.js`，文件开头需要保留 Node.js shebang：

```javascript
#!/usr/bin/env node

const fs = require('fs');
const execCmd = require('./tools/execCmd');
```

`package.json` 通过 `bin` 字段暴露命令：

```json
{
  "name": "aura_node_cli",
  "version": "1.0.0",
  "main": "src/index.js",
  "bin": {
    "aura-cli": "src/index.js"
  }
}
```

在工程目录执行以下命令即可将 `aura-cli` 链接到本机：

```shell
npm link
```
