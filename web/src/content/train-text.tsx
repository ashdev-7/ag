// Everything the training screen says, written for someone on their first
// day: no word is used before it is explained, nothing is hidden behind a
// click, and nothing is longer than it needs to be. Each claim matches what
// the code does (presets.ts, the engine).

import type { ReactNode } from 'react';

/** What the network is asked to do, per preset: the first thing a visitor reads. */
export const TRAIN_TASK: Record<string, { headline: string; task: ReactNode; lesson: ReactNode }> = {
  moons: {
    headline: 'Separate two interlocking moons',
    task: <>The network sees only a dot’s position and must say its colour. No straight line splits these shapes, so it has to learn a curve.</>,
    lesson: <>Two hidden layers are enough to bend the boundary around the moons.</>,
  },
  circles: {
    headline: 'Separate the inner circle from the outer ring',
    task: <>Blue dots in the middle, orange in a ring around them. The network sees only each dot’s position, so it must learn to draw a closed loop.</>,
    lesson: <>Each first-layer neuron can only cut the plane with a straight line (see its picture in ①); the next layer combines the cuts into a loop.</>,
  },
  spirals: {
    headline: 'Untangle two spirals',
    task: <>Two arms wound around each other: much harder than the moons, so this network has 32 neurons per hidden layer.</>,
    lesson: <>For a while nothing improves, then the boundary suddenly starts tracing the arms. Hard patterns need bigger networks and patience.</>,
  },
  xor: {
    headline: 'Solve the XOR puzzle',
    task: <>Opposite corners share a colour. No single straight line can separate them; this puzzle stalled neural-network research in the 1970s.</>,
    lesson: <>One hidden layer of 8 neurons draws two crossing lines, which a single neuron never could.</>,
  },
  vanishing: {
    headline: 'Watch a deep network fail: vanishing gradients',
    task: <>The moons again, through six layers of the old “sigmoid” bend. It will <b>not</b> learn, and that is the lesson.</>,
    lesson: <>Each sigmoid layer weakens the correction on its way back; by layer 1 it is about 10,000× too weak to change anything. Watch ④.</>,
  },
  'dead-relu': {
    headline: 'Watch a network that cannot learn: dead neurons',
    task: <>The circles again, but started so that every hidden neuron outputs zero. It will <b>not</b> learn, and that is the lesson.</>,
    lesson: <>A neuron stuck at zero sends no correction back, so nothing before it can change. ⑤ shows the zeros, ④ the empty corrections.</>,
  },
};

/** How training works, as three steps rather than a paragraph. */
export const TRAINING_LOOP = (batch: number): { name: string; line: ReactNode }[] => [
  { name: 'Guess', line: <>The <b>weights</b> start random, so the first answers are guesses.</> },
  { name: 'Measure', line: <>On {batch} dots, score how wrong it was: the <b>loss</b>.</> },
  { name: 'Nudge', line: <>Shift every weight slightly toward less wrong. Repeat thousands of times.</> },
];

export const OPTIMIZER_KEY: ReactNode = (
  <><b>Optimizer</b>: the rule for nudging. <b>lr</b>: nudge size (too big overshoots). <b>clip</b>: largest allowed nudge, 0 = none.</>
);

export const PANEL = (batch: number): Record<string, { title: string; explain: ReactNode }> => ({
  network: {
    title: 'The network',
    explain: <>
      Position goes in on the left, the answer comes out on the right. Each square is a <b>neuron</b>: it adds up its inputs, each times the
      <b> weight</b> on its line, and bends the result. <span className="k-pos">Orange</span> lines push up, <span className="k-neg">blue</span> push
      down, thicker push harder. Inside each square: what that neuron outputs across the whole plane.
    </>,
  },
  ops: {
    title: 'The network, as calculations',
    explain: <>The same network as the computer runs it: multiply by weights (@), add a bias (+), bend (tanh, relu), score (loss). Purple boxes are the weights being learned.</>,
  },
  boundary: {
    title: 'Its answer, everywhere',
    explain: <>The background is what the network answers at each point; the dots are the true data. The goal: every dot on its own colour.</>,
  },
  loss: {
    title: 'How wrong it is',
    explain: <>The <b>loss</b>. 0 is perfect; 0.69 is a coin flip. Solid: the {batch} dots of each step. Dashed: dots held back to test it, never trained on.</>,
  },
  grads: {
    title: 'How hard each layer is corrected',
    explain: <>The <b>gradient</b>: how much each layer’s weights are told to change, worked out backwards from the answer. A tiny bar means a layer barely learns.</>,
  },
  acts: {
    title: 'What each layer outputs',
    explain: <>The spread of each layer’s outputs over the current dots. Wide means neurons respond to position; a spike at zero means dead neurons.</>,
  },
  inspect: {
    title: 'Inside one layer',
    explain: <>Each cell is one weight, one line in ①: orange positive, blue negative. Beside it, the correction each weight receives.</>,
  },
});
