const test = require('node:test');
const assert = require('node:assert/strict');

const app = require('../index');

function request(server, path, method = 'GET') {
  const address = server.address();
  return fetch(`http://127.0.0.1:${address.port}${path}`, {method});
}

test('serves the root route', async (context) => {
  const server = app.listen(0);
  context.after(() => server.close());
  const response = await request(server, '/');
  assert.equal(response.status, 200);
  assert.equal(await response.text(), 'Hello World!');
});

test('serves the user PUT route', async (context) => {
  const server = app.listen(0);
  context.after(() => server.close());
  const response = await request(server, '/user', 'PUT');
  assert.equal(await response.text(), 'Got a PUT request at /user');
});
