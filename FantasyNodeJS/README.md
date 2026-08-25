# FantasyNodeJS

Node.js、Express、爬虫与 TypeScript 学习工程集合。每个子目录都是独立 npm 工程，必须在各自
目录安装依赖；不要在 `FantasyNodeJS` 根目录创建共享 `node_modules`。

| 子工程 | 用途 | 测试方式 |
|---|---|---|
| `AuraNodeCli` | 命令行参数处理 | Node 内置测试 |
| `AuraNodeSpider` | 可注入环境配置的爬虫入口 | 无数据库 smoke test |
| `MyMovieWeb` | Express 电影站示例 | 应用初始化测试 |
| `MyPicSpider`、`MySpiderDemo` | HTML 链接/图片解析 | 本地 HTML fixture |
| `NodeJsSamples/ExpressDemo` | 最小 Express 路由 | 路由单元测试 |
| `myExpressGenarator` | Express Generator 示例 | 应用与路由 smoke test |
| `NyxTSExpress` | TypeScript Express 完整示例 | Jest + Supertest |

推荐 Node.js 18 或更高版本。以任一子工程为例：

```bash
cd FantasyNodeJS/AuraNodeCli
npm ci
npm test
```

全部工程的离线测试可从仓库根目录执行：

```bash
for project in \
  FantasyNodeJS/AuraNodeCli \
  FantasyNodeJS/AuraNodeSpider \
  FantasyNodeJS/MyMovieWeb \
  FantasyNodeJS/MyPicSpider \
  FantasyNodeJS/MySpiderDemo \
  FantasyNodeJS/NodeJsSamples/ExpressDemo \
  FantasyNodeJS/myExpressGenarator \
  FantasyNodeJS/NyxTSExpress
do
  npm --prefix "$project" test
done
```

测试不得访问生产数据库或真实网站。爬虫规则使用本地 fixture 验证；网络抓取、MongoDB 和邮件
发送属于集成测试，需要单独的临时环境和非生产凭据。旧 Express/TypeScript 依赖会产生弃用警告，
升级时应逐个子工程处理并提交相应 `package-lock.json`，避免一次跨工程升级造成行为混淆。
