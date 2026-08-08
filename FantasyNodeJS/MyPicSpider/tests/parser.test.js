const test = require('node:test');
const assert = require('node:assert/strict');

const {analyzeMain} = require('../src/index');

test('extracts picture category links', async () => {
  const html = '<li class="left-list_li"><a href="/gallery">gallery</a></li>';
  assert.deepEqual(await analyzeMain(html), ['/gallery']);
});
