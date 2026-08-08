const test = require('node:test');
const assert = require('node:assert/strict');

const app = require('../src/app');

test('the Express application can start and answer HTTP requests', async (context) => {
  const server = app.listen(0);
  context.after(() => server.close());
  const {port} = server.address();
  const response = await fetch(`http://127.0.0.1:${port}/missing-route`);
  assert.equal(response.status, 404);
});
