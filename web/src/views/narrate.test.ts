import { describe, expect, it } from 'vitest';
import { accuracy, activationStory, boundaryStory, gradStory, lossStory, raceStory, trend } from './narrate.ts';

describe('live captions', () => {
  it('reads accuracy off the decision grid', () => {
    // a 2×2 grid over [-1, 1]²: right half says class 1
    const grid = { res: 2, data: [0, 1, 0, 1] };
    const data = { n: 4, nTrain: 2, xDim: 2, yDim: 2,
      x: Float32Array.from([0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5]),
      y: Float32Array.from([0, 1, 1, 0, 0, 1, 0, 1]) };
    // train: (0.5,0.5) class 1 → right; (−0.5,−0.5) class 0 → right. val: (0.5,−0.5) class 1 → right; (−0.5,0.5) class 1 → wrong
    expect(accuracy(grid, { x0: -1, x1: 1, y0: -1, y1: 1 }, data)).toEqual({ train: 1, val: 0.5 });
  });

  it('calls a trend only with enough points', () => {
    expect(trend([1, 0.9, 0.8])).toBe('early');
    expect(trend(Array.from({ length: 40 }, (_, i) => Math.exp(-i / 5)))).toBe('falling');
    expect(trend(Array.from({ length: 40 }, () => 0.69))).toBe('flat');
  });

  it('says when nothing has been trained, and compares the loss with a coin flip', () => {
    expect(lossStory([], [], 0)).toMatch(/Not trained yet/);
    expect(lossStory([0.7, 0.69, 0.7], [], 30)).toMatch(/coin flip/);
    expect(lossStory([0.7, 0.01], [], 30)).toMatch(/low/);
    expect(lossStory([0.7, 0.02], [0.5], 30)).toMatch(/memorising/);
    expect(boundaryStory(null, 0)).toMatch(/Not trained yet/);
    expect(boundaryStory({ train: 0.55, val: 0.5 }, 100)).toMatch(/close to guessing/);
  });

  it('names vanishing gradients, dead layers and healthy ones', () => {
    expect(gradStory([{ label: 'a', norm: 1e-5 }, { label: 'b', norm: 1e-1 }], [])).toMatch(/about 10,000× less/);
    expect(gradStory([{ label: 'a', norm: 0 }, { label: 'b', norm: 0 }], [])).toMatch(/cannot learn/);
    expect(gradStory([{ label: 'a', norm: 0.04 }, { label: 'b', norm: 0.01 }], [])).toMatch(/every layer is learning/);
    expect(activationStory([{ label: 'layer 1', fracZero: 1, std: 0 }], true)).toMatch(/dead/);
    expect(activationStory([{ label: 'layer 1', fracZero: 0.4, std: 0.5 }], true)).toMatch(/normal for ReLU/);
    expect(activationStory([{ label: 'layer 5', fracZero: 0, std: 0.06 }], false)).toMatch(/narrow band/);
    expect(activationStory([{ label: 'layer 1', fracZero: 0, std: 0.45 }], false)).toMatch(/vary widely/);
    expect(boundaryStory({ train: 0.53, val: 0.4 }, 900, 0.05)).toMatch(/almost one colour/);
  });

  it('narrates the race', () => {
    const flat = Array.from({ length: 80 }, () => 3.5);
    const fast = Array.from({ length: 80 }, (_, i) => Math.exp(-i / 5));
    const s = raceStory([{ name: 'sgd', loss: flat, diverged: false }, { name: 'adam', loss: fast, diverged: false }, { name: 'rmsprop', loss: [1], diverged: true }]);
    expect(s).toMatch(/Lowest so far: adam/);
    expect(s).toMatch(/sgd has almost stopped/);
    expect(s).toMatch(/rmsprop diverged/);
    const done = raceStory([{ name: 'sgd', loss: [3, 2, 1, -1], diverged: false }, { name: 'adam', loss: [3, -1, -1, -1], diverged: false }]);
    expect(done).toMatch(/All of them have reached the bottom/);
    expect(done).toMatch(/first to get there was adam/);
  });
});
