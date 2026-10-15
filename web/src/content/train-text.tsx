// Everything the training screen says, written for someone on their first
// day: no word is used before it is explained, and nothing is hidden behind
// a click. Each claim matches what the code does (presets.ts, the engine).

import type { ReactNode } from 'react';

/** What the network is asked to do, per preset: the first thing a visitor reads. */
export const TRAIN_TASK: Record<string, { headline: string; task: ReactNode; lesson: ReactNode }> = {
  moons: {
    headline: 'Teach a network to tell blue points from orange points',
    task: <>The dots below come in two colours, shaped like two interlocking half-moons. The network sees only each dot’s position and must learn to say which colour it is. No straight line can split these two shapes, so the network has to learn a curve.</>,
    lesson: <>What to notice: a network with two hidden layers can bend its boundary around a curved shape.</>,
  },
  circles: {
    headline: 'Teach a network to tell the inner circle from the outer ring',
    task: <>Blue dots sit in the middle, orange dots in a ring around them. The network sees only each dot’s position and must learn which colour it is: it has to draw a closed loop around the centre.</>,
    lesson: <>What to notice: each first-layer neuron can only cut the plane with one straight line (see its little picture); the next layer combines several cuts into a loop.</>,
  },
  spirals: {
    headline: 'Teach a network to untangle two spirals',
    task: <>Two spiral arms wind around each other, one blue, one orange. This is much harder than the moons: the network needs more neurons (32 per hidden layer) and more training before the boundary follows the arms.</>,
    lesson: <>What to notice: for a while almost nothing improves, then the boundary suddenly starts tracing the arms. Hard patterns need bigger networks and patience.</>,
  },
  xor: {
    headline: 'Teach a network the XOR pattern',
    task: <>Dots in the top-right and bottom-left corners are one colour; top-left and bottom-right the other. A single straight line can never separate these; this famous puzzle helped stall neural-network research in the 1970s. One hidden layer of 8 neurons solves it.</>,
    lesson: <>What to notice: the boundary becomes two crossing lines, something one neuron alone could never draw.</>,
  },
  vanishing: {
    headline: 'Watch a deep network fail to learn: vanishing gradients',
    task: <>The same two-moons task, but with six hidden layers that use the old “sigmoid” function. This network will <b>not</b> learn, and that is the point: watch why.</>,
    lesson: <>What to notice: the correction signal (the “gradient”) gets weaker every time it passes back through a sigmoid layer, so by the first layer it is about 10,000 times too weak to change anything. Panel ④ shows it shrinking layer by layer.</>,
  },
  'dead-relu': {
    headline: 'Watch a network that cannot learn at all: dead neurons',
    task: <>The circles task again, but the network was started with a bad setting: every hidden neuron outputs zero for every dot. This network will <b>not</b> learn, and that is the point.</>,
    lesson: <>What to notice: a neuron that always outputs zero sends no correction back, so nothing before it can change. Panel ⑤ shows every hidden neuron stuck at zero; panel ④ shows every correction is zero.</>,
  },
};

export const HOW_TRAINING_WORKS = (batch: number): ReactNode => (
  <>
    <b>How training works.</b> The network starts with random settings (its <b>weights</b>, the numbers on the lines in panel ①), so its first answers are guesses.
    Each <b>training step</b> it looks at a small group of {batch} dots, measures how wrong its answers were, and nudges every weight a little in the direction that makes it less wrong.
    Thousands of small nudges add up to a network that has learned the pattern. 20% of the dots are kept aside and never used for nudging: they test whether the network really learned the shape or only memorised the dots it practised on.
  </>
);

export const DO_THIS: ReactNode[] = [
  <>Press <b>Run</b>. Training starts; press it again (Pause) to stop.</>,
  <>Watch panel ②: the background colours are the network’s answers. They reshape until each dot sits on its own colour.</>,
  <>Watch panel ③: the line falls as the network gets less wrong.</>,
];

export const CONTROLS: ReactNode = (
  <><b>Run / Pause</b> trains continuously. <b>Step</b> does exactly one training step, so you can watch a single nudge. <b>Reset</b> starts over with the same random starting weights. <b>Speed</b>: how many steps happen between screen updates.</>
);

export const OPTIMIZER_INTRO: ReactNode = (
  <>The <b>optimizer</b> is the rule for nudging the weights. <b>lr</b> (learning rate) is the size of each nudge: too small and learning is slow, too large and it overshoots and blows up. <b>clip</b> caps how big one nudge can be (0 = no cap).</>
);

export const PANEL = (batch: number): Record<string, { title: string; explain: ReactNode }> => ({
  network: {
    title: '① The network: position in on the left, answer out on the right',
    explain: <>
      Each square is a <b>neuron</b>, a tiny calculator. The two on the left are the input: a dot’s horizontal (x₁) and vertical (x₂) position.
      The middle columns are <b>hidden layers</b>: each neuron adds up the numbers coming into it, each multiplied by the weight on its line, and passes the result through a simple bend.
      The two on the right are the answer: how sure the network is that the dot is blue, or orange.
      Each line is a <b>weight</b>: <span style={{ color: '#e66101' }}>orange</span> lines push their neuron up, <span style={{ color: '#2166ac' }}>blue</span> lines push it down, and thicker lines push harder. Training changes these lines.
      The picture inside each neuron shows what it outputs for every position on the plane: you can see each hidden neuron learn a simple shape, and later neurons combine them.
    </>,
  },
  ops: {
    title: '① The network, as the calculations the computer runs',
    explain: <>The same network written out step by step: multiply by the weights (@), add the bias (+), bend (tanh, relu…), and finally score how wrong it is (the loss). Purple boxes are the weights being learned. Each box shows the average of its numbers ± how spread out they are (blue), and its correction signal (orange ∂).</>,
  },
  boundary: {
    title: '② What the network answers for every point',
    explain: <>The background shows the network’s answer at every position: <b style={{ color: '#e66101' }}>orange</b> where it says “orange”, <b style={{ color: '#2166ac' }}>blue</b> where it says “blue”, and pale where it is unsure. The dots are the real data, in their true colours. When training works, every dot ends up on a background of its own colour. The small squares underneath are the same picture at earlier moments, so you can see how it changed.</>,
  },
  loss: {
    title: '③ How wrong is the network? (lower is better)',
    explain: <>This number is called the <b>loss</b>. 0 would mean perfectly confident and always right. The dotted line at 0.69 is what you get by guessing 50/50, so anything below it means the network has learned something. <b>Solid line:</b> the {batch} dots of the latest step (it jumps around because each group of dots is different). <b>Dashed line:</b> the dots kept aside for testing, which the network never trains on.</>,
  },
  grads: {
    title: '④ How strongly each layer is being corrected',
    explain: <>After each step, the network works out how much each weight should change to be less wrong. That correction signal is called the <b>gradient</b>, and it is computed from the answer backwards, one layer at a time. Each bar is how big the correction for one layer’s weights is right now (first layer at the top). A layer with a tiny bar is barely changing, so it is barely learning. The chart below shows the same thing over time.</>,
  },
  acts: {
    title: '⑤ What each layer’s neurons output',
    explain: <>For the current group of dots, a bar chart of every number each layer passes on: lowest value on the left, highest on the right, taller bar = more values there. Healthy layers give different outputs for different dots. If nearly everything sits at exactly zero, those neurons are “dead”: they output zero for every dot and can no longer learn.</>,
  },
  inspect: {
    title: '⑥ Inside one layer’s weights',
    explain: <>Every weight of one layer as a grid of coloured cells, one cell per line in panel ①: orange cells are positive weights, blue negative, white near zero. Next to it, each weight’s correction (gradient). Some optimizers also keep a memory per weight (a running average of past corrections), shown when there is one.</>,
  },
});
