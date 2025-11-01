AuraNodeCli的命令行工具开发

index.js的中需要加入

```
#!/usr/bin/env node

const fs = require('fs');
const execCmd = require('./tools/execCmd');
```

我们还需要在package.json中添加

```
{
  "name": "aura_node_cli",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "bin": {
    "aura-cli": "index.js"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}
```

​		我们怎么才能直接执行aura-cli的命令呢？

```
$ npm link
```

​		

​		