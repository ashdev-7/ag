import { describe, expect, it } from 'vitest';
import { SPEEDS, speedIndex } from './speed.ts';

describe('speed ladder', () => {
  it('rises steadily from 1 step a second', () => {
    expect(SPEEDS[0]).toBe(1);
    SPEEDS.slice(1).forEach((s, i) => expect(s).toBeGreaterThan(SPEEDS[i]));
  });
  it('finds every rung exactly, and the nearest rung for anything else', () => {
    SPEEDS.forEach((s, i) => expect(speedIndex(s)).toBe(i));
    expect(SPEEDS[speedIndex(12)]).toBe(10);
    expect(SPEEDS[speedIndex(40)]).toBe(50);
    expect(speedIndex(0.1)).toBe(0);
    expect(speedIndex(1e6)).toBe(SPEEDS.length - 1);
  });
});
