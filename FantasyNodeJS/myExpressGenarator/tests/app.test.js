const test = require('node:test');
const assert = require('node:assert/strict');

const app = require('../app');

test('serves the generated home page', async (context) => {
  const server = app.listen(0);
  context.after(() => server.close());
  const {port} = server.address();
  const response = await fetch(`http://127.0.0.1:${port}/`);
  assert.equal(response.status, 200);
  assert.match(await response.text(), /Express/);
});
