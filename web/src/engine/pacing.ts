// Training speed in steps per second, kept by the clock: the worker asks how
// many steps are due by now. Below the frame rate that is one step at a time
// with waits between; when frames are slow to draw, several steps catch up,
// so a heavy page trains at the speed it was asked for, not at its frame rate.

export interface Due {
  /** steps to take now (0: wait) */
  steps: number;
  /** when steps is 0: milliseconds until the next step is due */
  waitMs: number;
  /** more than a second behind: drop the debt rather than sprint to repay it */
  behind: boolean;
}

/**
 * Step k (from 0) is due at k / stepsPerSecond seconds after the start, so
 * the first one runs at once. At most a quarter second's worth of steps at a
 * time: enough to keep up with frames up to 250 ms apart, small enough that
 * catching up stays smooth.
 */
export function stepsDue(stepsPerSecond: number, elapsedMs: number, done: number): Due {
  const sps = Math.max(0.1, stepsPerSecond);
  const target = Math.floor((sps * Math.max(0, elapsedMs)) / 1000) + 1;
  const owed = target - done;
  if (owed <= 0) return { steps: 0, waitMs: (done / sps) * 1000 - elapsedMs, behind: false };
  const burst = Math.max(1, Math.ceil(sps / 4));
  return { steps: Math.min(owed, burst), waitMs: 0, behind: owed > Math.max(burst, sps) };
}

/** Share of the worker's time the extras of a frame (boundary, test loss) may take. */
export const EXTRA_SHARE = 0.2;

/**
 * Whether to recompute something that costs `costMs` and was last done at
 * `lastMs`: no more often than every `minMs`, and no more often than keeps it
 * to EXTRA_SHARE of the time. A big network's decision boundary takes tens of
 * milliseconds; recomputed every frame it would leave no time to train.
 */
export function refreshDue(nowMs: number, lastMs: number, minMs: number, costMs: number): boolean {
  return nowMs - lastMs >= Math.max(minMs, costMs / EXTRA_SHARE);
}
