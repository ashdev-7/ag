// Text saved through the wrong encoding turns ’ into â€™ and x₁ into xâ‚:
// the page would show it as is. No source file may contain such sequences.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const GARBLED = /â€|â‚|âˆ|â†|Ã[\u0080-¿]|�/;

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? files(p) : /\.(tsx?|css|html)$/.test(name) ? [p] : [];
  });
}

describe('source text', () => {
  it('has no garbled characters', () => {
    // this file quotes the patterns it looks for
    const bad = [...files('src'), 'index.html'].filter((f) => !f.endsWith('encoding.test.ts')).flatMap((f) =>
      readFileSync(f, 'utf8').split('\n').flatMap((line, i) => (GARBLED.test(line) ? [`${f}:${i + 1}`] : [])));
    expect(bad).toEqual([]);
  });

  it('has no byte-order marks', () => {
    expect([...files('src'), 'index.html'].filter((f) => readFileSync(f).subarray(0, 3).equals(Buffer.from([0xef, 0xbb, 0xbf])))).toEqual([]);
  });
});
