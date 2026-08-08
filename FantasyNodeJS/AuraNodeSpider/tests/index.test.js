const test = require('node:test');
const assert = require('node:assert/strict');

const {main} = require('../src/index');

test('main resolves without requiring a database connection', async () => {
  await assert.doesNotReject(main());
});
