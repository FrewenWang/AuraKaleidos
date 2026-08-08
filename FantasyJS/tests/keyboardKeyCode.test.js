import test from 'node:test';
import assert from 'node:assert/strict';

import * as keys from '../src/keyboardKeyCode.js';

test('exports standard keyboard key codes', () => {
  assert.equal(keys.KB_KEY_ENTER, 13);
  assert.equal(keys.KB_KEY_ArrowLeft, 37);
  assert.equal(keys.KB_KEY_Digit0, 48);
  assert.equal(keys.KB_KEY_KeyA, 65);
});
