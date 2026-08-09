const test = require('node:test');
const assert = require('node:assert/strict');

const app = require('../src/app');

test('the Express application initializes without opening a port', () => {
  assert.equal(typeof app, 'function');
  assert.equal(app.get('view engine'), 'jade');
  assert.match(app.get('views'), /views[\\/]pages$/);
  assert.ok(app._router.stack.length > 0);
});
