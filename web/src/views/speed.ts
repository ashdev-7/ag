// Training speed, in steps per second: a number a reader can picture. A
// slider moves through a fixed ladder of speeds, slow ones close together.

export const SPEEDS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000];

/** The ladder position nearest to `stepsPerSecond`. */
export function speedIndex(stepsPerSecond: number): number {
  let best = 0;
  SPEEDS.forEach((s, i) => {
    if (Math.abs(Math.log(s / stepsPerSecond)) < Math.abs(Math.log(SPEEDS[best] / stepsPerSecond))) best = i;
  });
  return best;
}
