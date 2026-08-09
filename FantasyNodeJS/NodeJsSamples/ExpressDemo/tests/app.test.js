const test = require('node:test');
const assert = require('node:assert/strict');

const {handlers} = require('../src/app');

function invoke(handler) {
  let body;
  handler({}, {send(value) { body = value; }});
  return body;
}

test('serves the root route', () => {
  assert.equal(invoke(handlers.getRoot), 'Hello World!');
});

test('serves the user PUT route', () => {
  assert.equal(invoke(handlers.putUser), 'Got a PUT request at /user');
});
