# My Pic Spider

使用内置 `fetch` 请求网页、使用 Cheerio 解析图片分类链接的教学项目。Node.js 版本要求 18 以上。

- `src/index.js`：请求、解析和命令入口。
- `tests/`：固定 HTML 的离线单元测试。
- `outputs/`：下载结果，默认忽略。

```bash
npm install
npm test
npm start
```
