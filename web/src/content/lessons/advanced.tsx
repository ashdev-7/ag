import { ActivationWidget } from '../../components/widgets.tsx';
import type { Lesson } from './types.ts';

export const ADVANCED: Lesson[] = [
  {
    id: 'momentum',
    level: 'advanced',
    title: 'Momentum: remembering where you were going',
    summary: 'Why plain gradient descent zig-zags in narrow valleys, and the heavy-ball fix.',
    question: <>Plain gradient descent works, but it is slow on some surfaces. Why, and how does adding “momentum” help?</>,
    sections: [
      {
        heading: 'The problem: narrow valleys',
        body: <>
          <p>Loss surfaces are rarely round bowls. More often they are long, narrow valleys: steep across, nearly flat along. The gradient points mostly <b>across</b> the valley, down the steep walls, and hardly at all along the floor toward the minimum. So gradient descent bounces from wall to wall and creeps forward.</p>
          <p>Lowering the learning rate stops the bouncing but makes the forward creep even slower. One step size cannot suit both directions.</p>
        </>,
      },
      {
        heading: 'The idea: a heavy ball',
        body: <>
          <p>Boris Polyak proposed in 1964 to think of a heavy ball rolling down the surface. It does not follow each local slope exactly; it keeps its velocity. Bounces across the valley cancel out over successive steps, while the small, consistent push along the floor adds up.</p>
          <p className="formula">v ← 0.9 · v + g        w ← w − lr · v</p>
          <p><code>v</code> is a running sum of past gradients, with old ones fading by a factor 0.9 per step. Consistent directions build up speed (up to about 10× the single gradient); alternating ones cancel.</p>
        </>,
      },
      {
        heading: 'Nesterov’s refinement',
        body: <p>Yurii Nesterov (1983) noticed it is better to evaluate the correction where the momentum is about to carry you, not where you are now: a look-ahead that brakes before overshooting. In this engine: <code>w ← w − lr · (g + 0.9 · v)</code>, the form PyTorch uses.</p>,
      },
      {
        heading: 'Where it is used',
        body: <p>SGD with momentum (typically 0.9) trained most of the landmark image networks of the 2010s and is still widely used. Adam, the next lesson, contains momentum as one of its two halves.</p>,
      },
    ],
    faq: [
      { q: 'Can momentum overshoot?', a: <>Yes. A heavy ball rolls past the bottom and back. That is the wobble you see on XOR. Usually it settles; with too large a learning rate it can diverge.</> },
      { q: 'Why 0.9?', a: <>A convention that works well in practice: it averages over roughly the last 10 steps. Values from 0.8 to 0.99 are common.</> },
      { q: 'What is the Rosenbrock function?', a: <>A test surface Howard Rosenbrock published in 1960 to compare optimisation methods: a curved valley that is easy to fall into and slow to follow, exactly the case momentum is meant for.</> },
    ],
    labs: [
      { preset: 'bowl', label: 'Zig-zag versus momentum', steps: [<>Press Race. <b>sgd</b> zig-zags across the steep direction; <b>momentum</b> smooths it out.</>] },
      { preset: 'rosenbrock', label: 'Following a curved valley', steps: [<>Press Race. All four reach the valley quickly; then compare how fast each moves along it in the loss chart.</>] },
    ],
  },

  {
    id: 'adaptive',
    level: 'advanced',
    title: 'Adaptive optimizers: AdaGrad, RMSProp, Adam, AdamW',
    summary: 'A separate step size for every parameter, learned from the gradients themselves.',
    question: <>Different parameters see gradients of very different sizes. Can each one get its own learning rate, automatically?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>One global learning rate must be small enough for the parameter with the steepest slope, which makes it far too small for parameters with gentle slopes. In a network, some weights get large, frequent gradients and others small, rare ones (a rarely-seen word in a language model, for example).</p>,
      },
      {
        heading: 'AdaGrad (2011)',
        body: <p>John Duchi, Elad Hazan and Yoram Singer divided each parameter’s step by the square root of the <b>sum of all its past squared gradients</b>. Steep, frequent directions get smaller steps; rare ones keep larger steps. The flaw: the sum only grows, so steps shrink toward zero and learning eventually stops.</p>,
      },
      {
        heading: 'RMSProp (2012)',
        body: <>
          <p>Geoffrey Hinton, in a lecture of his online course rather than a paper, replaced the sum with a <b>running average</b> that forgets old gradients:</p>
          <p className="formula">s ← β · s + (1 − β) · g²        w ← w − lr · g / (√s + ε)</p>
          <p>The step is now roughly <code>lr</code> in every direction, whatever the raw gradient size. Steep and shallow directions move at similar speeds, and learning never grinds to a halt.</p>
        </>,
      },
      {
        heading: 'Adam (2014)',
        body: <>
          <p>Diederik Kingma and Jimmy Ba combined momentum (a running average of gradients, <code>m</code>) with RMSProp’s scaling (a running average of squared gradients, <code>v</code>):</p>
          <p className="formula">m ← 0.9 m + 0.1 g     v ← 0.999 v + 0.001 g²     w ← w − lr · m̂ / (√v̂ + ε)</p>
          <p>The hats are a <b>bias correction</b>: <code>m</code> and <code>v</code> start at zero, so early on they underestimate; dividing by <code>1 − 0.9ᵗ</code> and <code>1 − 0.999ᵗ</code> fixes that. Adam works well with little tuning, which made it the default for a decade.</p>
        </>,
      },
      {
        heading: 'AdamW (2017)',
        body: <p>Weight decay (shrinking weights slightly every step, to discourage large weights) interacts badly with Adam’s per-parameter scaling when added to the gradient. Ilya Loshchilov and Frank Hutter applied it directly to the weights instead. AdamW is the standard optimizer for training large language models. (On this site weight decay is 0, so AdamW behaves like Adam.)</p>,
      },
      {
        heading: 'Why it helps at saddle points',
        body: <p>In high dimensions, most flat places are <b>saddles</b> (downhill in some directions, uphill in others), not minima. Near a saddle the gradient is tiny, so plain SGD crawls. Because adaptive methods divide by the gradient’s recent size, a tiny but consistent gradient still produces a full-sized step, and they escape quickly.</p>,
      },
    ],
    faq: [
      { q: 'Is Adam always better than SGD?', a: <>No. It is easier to get working, but carefully tuned SGD with momentum sometimes generalises better on image tasks. Many teams try both.</> },
      { q: 'What is ε for?', a: <>A tiny number (1e-8) that prevents division by zero when a gradient has been zero for a while.</> },
      { q: 'Why does the race show RMSProp and Adam taking similar-sized steps everywhere?', a: <>Because they divide the gradient by its own recent size, the step length is about <code>lr</code> regardless of how steep the surface is.</> },
    ],
    labs: [
      { preset: 'saddle', label: 'Escaping a saddle', steps: [<>Press Race. <b>sgd</b> stalls near the saddle at (0, 0) (the flat stretch of its loss curve); <b>rmsprop</b> and <b>adam</b> escape almost immediately.</>] },
      { preset: 'moons', label: 'Adam against SGD on a network', steps: [<>Tick <i>compare two optimizers</i>, then Run: same starting weights, two optimizers.</>, <>Click a weight box and look at its first and second moment: Adam’s <code>m</code> and <code>v</code>.</>] },
    ],
  },

  {
    id: 'vanishing',
    level: 'advanced',
    title: 'Vanishing and exploding gradients',
    summary: 'Why deep networks were nearly untrainable for decades, and what fixed it.',
    question: <>If more layers can represent more, why were deep networks so hard to train until around 2010?</>,
    sections: [
      {
        heading: 'The problem',
        body: <>
          <p>Backpropagation multiplies by one local derivative per layer on the way back. Through 10 layers, the gradient reaching the first layer is a product of 10 factors. If each factor is below 1, the product shrinks exponentially: the gradient <b>vanishes</b>, and the early layers barely learn. If each is above 1, it grows exponentially: it <b>explodes</b>, and training blows up.</p>
          <p>Sepp Hochreiter analysed this in his 1991 thesis, and Yoshua Bengio, Patrice Simard and Paolo Frasconi in 1994, especially for recurrent networks, which apply the same layer many times in a row.</p>
        </>,
      },
      {
        heading: 'Why sigmoid makes it worse',
        body: <>
          <p>The sigmoid’s slope is at most 0.25 (at 0) and nearly 0 for large inputs. Six sigmoid layers multiply the gradient by at most <code>0.25⁶ ≈ 0.00024</code>, and usually much less. Look at the dashed slope line below for sigmoid, then tanh (at most 1) and ReLU (exactly 1 where active).</p>
          <ActivationWidget initial="sigmoid" />
        </>,
      },
      {
        heading: 'What fixed it',
        body: <ul>
          <li><b>ReLU</b> (around 2010–2011): slope exactly 1 where active, so it does not shrink the gradient.</li>
          <li><b>Careful initialisation</b> (2010, 2015): starting weights scaled so signals and gradients keep roughly the same size through every layer (next lesson).</li>
          <li><b>LSTMs</b> (1997): for recurrent networks, a memory path that the gradient can travel along without being multiplied down.</li>
          <li><b>Normalisation</b> (batch norm, 2015) and <b>residual connections</b> (ResNet, 2015): skip paths that let the gradient bypass layers, making networks of 100+ layers trainable.</li>
          <li><b>Gradient clipping</b> (Razvan Pascanu, Tomas Mikolov and Yoshua Bengio, 2013) for explosions: if the gradient is longer than a threshold, scale it down to that length. That is the <i>clip</i> setting on this site.</li>
        </ul>,
      },
    ],
    faq: [
      { q: 'Why not just use a huge learning rate for the early layers?', a: <>The shrinkage depends on the current inputs and changes every step, by orders of magnitude; no fixed rate per layer can track it. Adaptive optimizers help partly for this reason.</> },
      { q: 'Does clipping change the direction of the step?', a: <>No, only its length. All gradients are scaled by the same factor.</> },
      { q: 'Why is the loss stuck at 0.69 in the vanishing example?', a: <>0.69 is a coin flip. The last layer can learn a little, but it sits on top of five layers that barely change, and these happen to produce almost no useful signal.</> },
    ],
    labs: [{
      preset: 'vanishing', label: 'Six sigmoid layers',
      steps: [<>Press Run, then read <i>Gradient norm per layer</i>: about four orders of magnitude between the first layer (dark) and the output (orange).</>, <>Compare with the <i>Moons</i> lab, the same data with two tanh layers: there the layers’ gradient norms stay within about a factor of 4 of each other, and the loss falls to near 0.</>],
    }],
  },

  {
    id: 'relu-init',
    level: 'advanced',
    title: 'ReLU, dead neurons and initialisation',
    summary: 'The activation that made deep learning work, its failure mode, and how starting weights are chosen.',
    question: <>ReLU fixed vanishing gradients. Does it have its own problems, and how should the starting weights be chosen?</>,
    sections: [
      {
        heading: 'ReLU and the dead neuron',
        body: <>
          <p><code>ReLU(x) = max(0, x)</code> passes gradients through untouched when its input is positive. When the input is negative, its output and its slope are both exactly 0. A neuron whose input is negative for <b>every</b> example outputs 0 forever, sends no gradient back, and so its weights never change: it is <b>dead</b>.</p>
          <p>Neurons die from a bad start (biases too negative) or from one large update that pushes them into the negative region. Leaky ReLU (2013) keeps a small slope, e.g. 0.01, for negative inputs so a neuron can always recover.</p>
        </>,
      },
      {
        heading: 'Why the starting weights matter',
        body: <>
          <p>If starting weights are too small, signals shrink layer by layer toward zero; too large, they grow and saturate. Either way gradients vanish or explode before training has begun. The goal is to keep the spread of values roughly constant from layer to layer.</p>
          <ul>
            <li><b>Xavier / Glorot</b> (Xavier Glorot and Yoshua Bengio, 2010), for tanh and sigmoid: random weights with standard deviation <code>√(2 / (fan_in + fan_out))</code>, where fan_in and fan_out are the layer’s numbers of inputs and outputs.</li>
            <li><b>He / Kaiming</b> (Kaiming He and colleagues, 2015), for ReLU: <code>√(2 / fan_in)</code>. The extra factor of 2 compensates for ReLU zeroing out half its inputs.</li>
          </ul>
          <p>Biases usually start at 0. The examples on this site use He for ReLU networks and Xavier for the rest.</p>
        </>,
      },
      {
        heading: 'Why weights start random at all',
        body: <p>If every weight in a layer started equal, every neuron in it would compute the same thing and receive the same gradient, and they would stay identical forever. Randomness <b>breaks the symmetry</b> so each neuron can learn something different.</p>,
      },
    ],
    faq: [
      { q: 'Is a few dead neurons a problem?', a: <>Not necessarily: a network has many, and some ReLUs being off for some inputs is how ReLU networks work (46% zeros is normal on Circles). It is a problem when most units are dead for all inputs.</> },
      { q: 'Can a dead neuron come back?', a: <>Only if the layers before it change enough to make its input positive for some example. If they are dead too, nothing changes, as in the Dead ReLUs example.</> },
      { q: 'Why are the biases −1.5 in that example?', a: <>To make the failure happen on purpose: with small weights, every input to every hidden ReLU is negative from the very first step.</> },
    ],
    labs: [
      { preset: 'dead-relu', label: 'A network that cannot learn', steps: [<>Press Run: both hidden layers are 100% zero, every weight gradient is 0, and the loss stays at 0.69.</>] },
      { preset: 'circles', label: 'The same task, healthy start', steps: [<>Press Run: the same data and network with He initialisation and zero biases learns in a few hundred steps, with about half of each layer’s outputs zero, as normal for ReLU.</>] },
    ],
  },

  {
    id: 'capacity',
    level: 'advanced',
    title: 'Capacity: how big should a network be?',
    summary: 'Width, depth and what a harder problem needs.',
    question: <>How many layers and neurons does a problem need, and what happens with too few?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>The structure of a network is chosen by a person before training. Too small, and no setting of its knobs can represent the pattern: the loss stops falling at a high value (<b>underfitting</b>). Too large, and it trains slowly and can memorise instead of learning (overfitting), though modern practice shows large networks often generalise surprisingly well with enough data.</p>,
      },
      {
        heading: 'Width and depth',
        body: <>
          <p><b>Width</b> (neurons per layer) is how many pieces a layer can cut the input into. <b>Depth</b> (number of layers) is how many times pieces can be combined. Depth is often more efficient: some patterns need exponentially many neurons in one wide layer but far fewer spread over several layers, because each layer reuses the features of the one before.</p>
          <p>The examples here show the range: XOR needs 8 hidden units; Moons, two layers of 8; Spirals, whose arms wind almost twice around the centre, two layers of 32.</p>
        </>,
      },
      {
        heading: 'How it is chosen in practice',
        body: <p>Start from a structure known to work on similar problems, make it bigger until the training loss gets low, then watch the validation loss for overfitting. Architectures such as convolutional networks and transformers build in assumptions about the data (images, sequences) so that less capacity is wasted.</p>,
      },
    ],
    faq: [
      { q: 'Why does the Spirals loss sit flat before falling?', a: <>Early on, the network has not found any feature that follows the arms; the boundary is roughly a line and the loss stays near a coin flip. Once a few units align with the spiral, progress is rapid.</> },
      { q: 'Is bigger always better?', a: <>With enough data and compute, larger models have tended to do better, which is the story of recent language models. With little data, a smaller model or other forms of regularisation are safer.</> },
    ],
    labs: [{
      preset: 'spirals', label: 'A harder pattern',
      steps: [<>Press Run and watch the boundary trace the arms turn by turn; compare the loss curve’s flat start with Moons.</>],
    }],
  },
];
