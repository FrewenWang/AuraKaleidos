const test = require('node:test');
const assert = require('node:assert/strict');

const {analyzeMain} = require('../src/index');

test('extracts links from supported sections', async () => {
  const html = `
    <ul><li class="left-list_li"><a href="/left">left</a></li></ul>
    <div class="hot public-box"><a href="/hot">hot</a></div>
    <div class="channel public-box"><a href="/channel">channel</a></div>`;
  assert.deepEqual(await analyzeMain(html), ['/left', '/hot', '/channel']);
});
