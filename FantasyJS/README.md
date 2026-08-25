# FantasyJS

浏览器 JavaScript 学习示例。当前自动测试覆盖键盘按键码导出，其他文件可通过 `index.html`
在浏览器中查看。推荐使用 Node.js 18 或更高版本。

```bash
cd FantasyJS
npm ci
npm test
```

`npm test` 使用 Node 内置测试运行器，不启动浏览器和服务。依赖版本由 `package-lock.json` 锁定；
修改 `package.json` 后应同步提交 lockfile。`node_modules/` 和本地日志不得纳入 Git。

```text
src/                         JavaScript 示例源码
tests/keyboardKeyCode.test.js  离线单元测试
index.html                   浏览器入口
```

