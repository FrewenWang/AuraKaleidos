const test = require('node:test');
const assert = require('node:assert/strict');

const execCmd = require('../src/tools/execCmd');

test('help flags are recognized', () => {
  assert.equal(execCmd.needShowCliHelp([]), true);
  assert.equal(execCmd.needShowCliHelp(['--help']), true);
  assert.equal(execCmd.needShowCliHelp(['build']), false);
});

test('execCmd strips the node executable and script path', () => {
  const result = execCmd(['node', 'aura-cli', 'build']);
  assert.deepEqual(result, {args: ['build'], showHelp: false});
});

test('execCmd invokes its callback', () => {
  let callbackResult;
  execCmd(['node', 'aura-cli', '-h'], result => { callbackResult = result; });
  assert.equal(callbackResult.showHelp, true);
});
