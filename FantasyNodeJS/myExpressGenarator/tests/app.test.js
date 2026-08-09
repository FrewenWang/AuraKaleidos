const test = require('node:test');
const assert = require('node:assert/strict');

const app = require('../app');

test('initializes the generated application and routes', () => {
  assert.equal(typeof app, 'function');
  assert.equal(app.get('view engine'), 'pug');
  assert.match(app.get('views'), /views$/);
  assert.ok(app._router.stack.length > 0);
});
