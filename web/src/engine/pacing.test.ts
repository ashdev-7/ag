import { describe, expect, it } from 'vitest';
import { refreshDue, stepsDue } from './pacing.ts';

/** Simulate the worker: ticks whenever a frame is acknowledged `frameMs` after it was posted. */
function simulate(sps: number, frameMs: number, seconds: number): number {
  let t = 0, done = 0, t0 = 0;
  while (t < seconds * 1000) {
    const d = stepsDue(sps, t - t0, done);
    if (d.steps === 0) { t += Math.max(1, d.waitMs); continue; }
    if (d.behind) t0 = t - ((done + d.steps) / sps) * 1000;
    done += d.steps;
    t += frameMs;   // drawing the frame, then the ack
  }
  return done;
}

describe('stepsDue', () => {
  it('takes the first step at once', () => {
    expect(stepsDue(1, 0, 0).steps).toBe(1);
  });
  it('then waits until the next step is due', () => {
    const d = stepsDue(2, 100, 1);
    expect(d.steps).toBe(0);
    expect(d.waitMs).toBe(400);
  });
  it('keeps the asked speed when frames are fast', () => {
    for (const sps of [1, 5, 20, 100, 1000]) expect(Math.abs(simulate(sps, 5, 10) - sps * 10)).toBeLessThanOrEqual(Math.max(2, sps * 0.03));
  });
  it('keeps the asked speed when frames are slow, by taking several steps a frame', () => {
    // 20 steps a second with 120 ms frames: one step per frame would give only 8
    expect(simulate(20, 120, 10)).toBeGreaterThanOrEqual(195);
    expect(simulate(100, 120, 10)).toBeGreaterThanOrEqual(970);
  });
  it('never runs ahead of the clock', () => {
    for (const sps of [1, 5, 20, 100]) expect(simulate(sps, 1, 5)).toBeLessThanOrEqual(sps * 5 + 1);
  });
  it('catches up in small bites, and gives up a debt of over a second', () => {
    expect(stepsDue(20, 1000, 1).steps).toBe(5);        // 20 owed, 5 at a time (1/4 s worth)
    expect(stepsDue(20, 1000, 1).behind).toBe(false);
    expect(stepsDue(20, 3000, 1).behind).toBe(true);
    expect(stepsDue(1000, 1000, 1).steps).toBe(250);
  });
});

describe('refreshDue', () => {
  it('refreshes a cheap extra as often as asked', () => {
    expect(refreshDue(1033, 1000, 33, 2)).toBe(true);
    expect(refreshDue(1020, 1000, 33, 2)).toBe(false);
  });
  it('spaces out an expensive one so it takes at most a fifth of the time', () => {
    // 40 ms each: at most every 200 ms
    expect(refreshDue(1100, 1000, 33, 40)).toBe(false);
    expect(refreshDue(1200, 1000, 33, 40)).toBe(true);
  });
  it('always does the first one', () => {
    expect(refreshDue(5, -Infinity, 33, 1000)).toBe(true);
  });
});
