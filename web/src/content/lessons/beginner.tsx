import { DescentWidget, SlopeWidget } from '../../components/widgets.tsx';
import type { Lesson } from './types.ts';

export const BEGINNER: Lesson[] = [
  {
    id: 'network',
    level: 'beginner',
    title: 'What a neural network is',
    summary: 'A formula with adjustable knobs, and why that is enough to learn from examples.',
    question: <>Everyone says neural networks “learn”. What is actually inside one, and what does learning change?</>,
    sections: [
      {
        heading: 'The problem',
        body: <>
          <p>Some rules are easy to write down: a tax rate, a unit conversion. Others are not. Nobody can write the rule that tells a cat photo from a dog photo, or turns speech into text, as a list of if-statements. But we do have <b>examples</b>: thousands of photos with the right answer attached.</p>
          <p>So the question becomes: can we write a formula with blanks in it, and let the examples fill in the blanks?</p>
        </>,
      },
      {
        heading: 'Before neural networks',
        body: <>
          <p>Statisticians already did this for simple cases. A straight line <code>y = w·x + b</code> has two blanks, <code>w</code> and <code>b</code>; “fitting a line” means choosing them so the line passes close to the data points. That works when the relationship really is a straight line, and fails when it is anything more tangled.</p>
          <p>In 1943 Warren McCulloch and Walter Pitts described an “artificial neuron”: add up weighted inputs and fire if the sum is large enough. In 1958 Frank Rosenblatt built the <i>perceptron</i>, a machine that adjusted those weights from examples. Modern networks are still made of the same piece.</p>
        </>,
      },
      {
        heading: 'The idea',
        body: <>
          <p>A <b>neuron</b> does three things: multiply each input by a weight, add a bias, and pass the result through a simple bending function (the <i>activation</i>, e.g. <code>tanh</code>):</p>
          <p className="formula">out = tanh(x₁·w₁ + x₂·w₂ + b)</p>
          <p>The weights <code>w</code> and the bias <code>b</code> are the knobs, called <b>parameters</b>. A <b>network</b> is many neurons side by side (a <i>layer</i>) and layers one after another, each layer’s outputs feeding the next layer’s inputs. The whole thing is still just one big formula with many knobs.</p>
          <p><b>Learning</b> means turning the knobs until the formula gives the right answers on the examples. Nothing else changes: the structure stays fixed, only the numbers in it move.</p>
        </>,
      },
      {
        heading: 'Why it works',
        body: <>
          <p>Each neuron on its own is weak: roughly a soft on/off switch along one straight line. But stacking enough of them can approximate almost any reasonable function; this was proved in 1989 (George Cybenko) and is called the <i>universal approximation theorem</i>. It says a good setting of the knobs exists. It does not say how to find it; the rest of this course is about finding it.</p>
        </>,
      },
      {
        heading: 'Where it is used',
        body: <p>Every modern neural network, from image recognition to language models, is this same idea at a larger scale: a fixed structure of simple operations, with parameters set by learning. Large language models have hundreds of billions of parameters; the examples on this site have between 2 parameters (the optimizer races) and about 1,200 (the spirals network).</p>,
      },
    ],
    faq: [
      { q: 'Is it really like a brain?', a: <>Only loosely. The name comes from the 1940s inspiration. A real neuron is far more complex, and brains are not known to learn the way these networks do. Think of it as a flexible formula, not a model of biology.</> },
      { q: 'Why is it called a “weight”?', a: <>Because it sets how much weight an input carries: a large weight makes that input matter a lot, zero makes it ignored, a negative one makes it count against.</> },
      { q: 'What does the bias do?', a: <>It shifts the neuron’s threshold. Without it, a neuron with all-zero inputs would always output the same value, and every decision line would have to pass through the origin.</> },
      { q: 'Who decides the structure (how many layers, how many neurons)?', a: <>A person. The structure is a design choice; only the parameters are learned. Choosing the structure is covered in the last lesson, on capacity.</> },
    ],
    labs: [{
      preset: 'neuron', hash: 'pass=forward', label: 'One neuron, forward',
      steps: [<>Press ▶ (one step) to compute the boxes one at a time: <code>x·w</code>, then <code>z</code> (adds <code>b</code>), then <code>out</code> (the tanh of <code>z</code>).</>, <>The blue numbers are values. <code>w</code> and <code>b</code> (violet boxes) are the knobs; <code>x</code> (grey box) is the input.</>],
    }],
  },

  {
    id: 'loss',
    level: 'beginner',
    title: 'Measuring wrongness: the loss',
    summary: 'Turning “how wrong is it?” into one number a computer can make smaller.',
    question: <>To turn the knobs in the right direction, we first need to say how wrong the network is. How do you turn “wrong” into a number?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>A network with random knobs gives wrong answers. To improve it we must compare two settings and say which is better, automatically, millions of times. That needs a single number, where smaller means better: the <b>loss</b> (also called cost or error).</p>,
      },
      {
        heading: 'What came before, and why it was not enough',
        body: <>
          <p>The obvious score is “how many answers were wrong”. It is useless for learning: a tiny turn of a knob almost never changes the count, so it gives no hint which way to turn. It is flat almost everywhere and then jumps.</p>
          <p>Around 1805 Adrien-Marie Legendre, fitting orbits of comets, used the <b>sum of squared errors</b>: for each example, square the difference between prediction and truth, and add them up. Carl Friedrich Gauss used the same method. It is still the standard for predicting numbers.</p>
        </>,
      },
      {
        heading: 'How it works',
        body: <>
          <p>The <b>mean squared error</b> over N examples:</p>
          <p className="formula">L = ( (p₁ − t₁)² + (p₂ − t₂)² + … + (p_N − t_N)² ) / N</p>
          <p>where <code>p</code> is the prediction and <code>t</code> the target. Exactly right gives 0; the further off, the larger it gets.</p>
          <p>For choosing between classes (cat or dog) a different loss, <b>cross-entropy</b>, works better; it has its own lesson.</p>
        </>,
      },
      {
        heading: 'Why squared?',
        body: <>
          <ul>
            <li><b>Smooth:</b> a small change in the knobs makes a small change in the loss, so the loss can tell us which direction is better. Counting mistakes cannot.</li>
            <li><b>Big errors count more:</b> being off by 2 costs 4, off by 10 costs 100. The network is pushed hardest where it is most wrong.</li>
            <li><b>Simple slope:</b> the derivative of <code>(p − t)²</code> is <code>2(p − t)</code>, the error itself, times 2. You will see this number in the labs.</li>
          </ul>
        </>,
      },
      {
        heading: 'Where it is used',
        body: <p>Every training run minimises a loss. Squared error is used whenever the output is a number (prices, temperatures, positions); cross-entropy whenever it is a choice (classes, the next word in a sentence). The loss is the only thing the network is ever told about what “good” means.</p>,
      },
    ],
    faq: [
      { q: 'Why one number and not one per example?', a: <>To compare two settings of the knobs you need one score. Averaging over examples also makes the loss independent of how many examples there are.</> },
      { q: 'Is a low loss the same as a good network?', a: <>Not necessarily. A network can get a low loss on its training examples by memorising them and still fail on new ones. That is <i>overfitting</i>, covered in the training-loop lesson.</> },
      { q: 'Why not the absolute error |p − t|?', a: <>It is also used (it cares less about outliers), but its slope jumps from −1 to +1 at zero, which makes the last steps of learning jittery. Squared error’s slope shrinks smoothly as the error shrinks.</> },
    ],
    labs: [{
      preset: 'neuron', hash: 'pass=forward', label: 'The loss of one neuron',
      steps: [<>Press ⏭ (jump to the end). The last box, <code>loss</code>, is <code>(out − 0.5)²</code>: how far the neuron’s output is from the target 0.5, squared.</>, <>Note it down: in the gradient-descent lesson you will watch it shrink.</>],
    }],
  },

  {
    id: 'derivatives',
    level: 'beginner',
    title: 'Which way is downhill: derivatives',
    summary: 'For every knob, how much the loss changes when you nudge it.',
    question: <>We have a number to make smaller. With thousands of knobs, how do we know which way to turn each one?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>Picture standing on a hillside in fog. You cannot see the valley, but you can feel the slope under your feet. If you know which way is downhill, you can take a step. For a network, the “position” is the setting of all its knobs and the “height” is the loss. We need the slope.</p>,
      },
      {
        heading: 'The simplest way: nudge and measure',
        body: <>
          <p>Nudge one knob by a tiny amount <code>h</code>, recompute the loss, and see how much it changed:</p>
          <p className="formula">slope ≈ ( L(w + h) − L(w) ) / h</p>
          <p>This is a <b>finite difference</b>. As <code>h</code> gets smaller, this becomes exactly the <b>derivative</b> <code>dL/dw</code>, the idea Newton and Leibniz formalised in the 1600s. Move the sliders below and watch the two slopes.</p>
          <SlopeWidget />
        </>,
      },
      {
        heading: 'Many knobs: the gradient',
        body: <>
          <p>With many knobs, take the derivative for each one separately, holding the others still: these are <b>partial derivatives</b>, written <code>∂L/∂w</code>. Together they form the <b>gradient</b>: a list with one number per knob.</p>
          <ul>
            <li>The <b>sign</b> says which way: positive means turning the knob up makes the loss worse, so turn it down.</li>
            <li>The <b>size</b> says how much it matters: a large gradient means this knob has a big effect right now.</li>
          </ul>
          <p>On this site, gradients are the orange numbers marked <code>∂</code>.</p>
        </>,
      },
      {
        heading: 'Why not just nudge every knob?',
        body: <p>It works, and the tests of this site’s engine do exactly that to check its answers. But it needs one extra run of the whole network per knob. A network with a million knobs would need a million runs to take one step. That cost is the problem backpropagation solves, two lessons from now.</p>,
      },
    ],
    faq: [
      { q: 'What does ∂L/∂a = 4 actually mean?', a: <>If you increase <code>a</code> by a tiny amount, the loss increases by about 4 times that amount. In <code>d = a·b + a</code> with <code>a = 2, b = 3</code>: raising <code>a</code> to 2.01 makes <code>d</code> 8.04 instead of 8, a change of 0.04 = 4 × 0.01.</> },
      { q: 'Why “∂” and not “d”?', a: <>The curly ∂ means “partial”: the derivative with respect to one input while the others are held fixed.</> },
      { q: 'Is the slope the same everywhere?', a: <>No. It is the slope <i>here</i>, at the current knob settings. After a step it must be computed again; that is why training repeats.</> },
    ],
    labs: [{
      preset: 'expression', hash: 'pass=reverse', label: 'Read gradients off a graph',
      steps: [<>Press ⏭ (jump to the end). <code>∂d/∂a = 4</code> and <code>∂d/∂b = 2</code>.</>, <>Check by hand: nudging <code>b</code> from 3 to 3.01 changes <code>d = a·b + a</code> by 0.02 = 2 × 0.01.</>],
    }],
  },

  {
    id: 'descent',
    level: 'beginner',
    title: 'Walking downhill: gradient descent',
    summary: 'Take a small step against the gradient; repeat. How big a step, and what goes wrong.',
    question: <>We know which way is downhill for every knob. How far should we step, and is that really how networks learn?</>,
    sections: [
      {
        heading: 'Before this',
        body: <p>Augustin-Louis Cauchy described this method in 1847, to solve equations in astronomy: move against the slope, recompute, repeat. It needs no clever insight into the problem, only the gradient, which is exactly why it suits neural networks.</p>,
      },
      {
        heading: 'How it works',
        body: <>
          <p className="formula">w ← w − lr · ∂L/∂w</p>
          <p>For every knob at once: subtract its gradient times a small number, the <b>learning rate</b> (<code>lr</code>). Then compute the gradient again at the new point and repeat. Each repetition is one <b>training step</b>.</p>
          <p>The minus sign is the whole idea: the gradient points uphill, so we go the other way.</p>
          <DescentWidget />
        </>,
      },
      {
        heading: 'The learning rate',
        body: <ul>
          <li><b>Too small:</b> correct but painfully slow; thousands of tiny steps.</li>
          <li><b>About right:</b> the loss falls quickly and settles.</li>
          <li><b>Too large:</b> each step overshoots the bottom, lands higher on the other side, and the loss bounces or explodes. On this site an exploded run stops with “nan” (not a number).</li>
        </ul>,
      },
      {
        heading: 'Why it works',
        body: <p>Close up, any smooth surface looks like a tilted plane, and on a plane the gradient direction is exactly the steepest way down. A small enough step therefore always lowers the loss. Repeating it walks to a point where the slope is zero: a bottom, at least locally.</p>,
      },
      {
        heading: 'Where it is used',
        body: <p>Every neural network is trained with gradient descent or a refinement of it (momentum, Adam: see the advanced lessons). Even models with billions of parameters take exactly this kind of step, millions of times.</p>,
      },
    ],
    faq: [
      { q: 'Does it find the best possible knob settings?', a: <>Not guaranteed. It finds a point where the slope is zero, which could be a local dip or a flat saddle. Surprisingly, for large networks the dips it finds are usually good enough; why this works so well is still an active research question.</> },
      { q: 'How do people choose the learning rate?', a: <>By trying several, usually spaced by factors of 3 or 10, and keeping the one whose loss falls fastest without bouncing. Many also lower it during training (a <i>schedule</i>).</> },
      { q: 'Why not take one big exact jump to the bottom?', a: <>For a simple bowl you could solve for the bottom directly (that is what “least squares” formulas do). For a network the surface is too complicated to solve; we only ever know the local slope.</> },
    ],
    labs: [
      {
        preset: 'bowl', label: 'Step size in 2-D',
        steps: [<>Press Race and watch <b>sgd</b> (orange): its steps bounce across the narrow direction because the learning rate is near the limit for that direction.</>, <>Set its lr to 0.11 and press Reset, then Race: now each step overshoots further than the last, its loss climbs instead of falling, and the race marks it diverged.</>],
      },
      {
        preset: 'neuron', hash: 'pass=train', label: 'Training steps on one neuron',
        steps: [<>Press ⏭ (jump to the end): the last steps update <code>w</code> and <code>b</code>.</>, <>Press <i>Next train step</i> a few times and watch the value in the <code>loss</code> box fall.</>],
      },
    ],
  },

  {
    id: 'chain-rule',
    level: 'beginner',
    title: 'The chain rule and computation graphs',
    summary: 'Break a big formula into small steps; each step knows its own slope.',
    question: <>A network is one enormous formula. How can anything compute its derivative without a mathematician doing it by hand?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>Deriving the gradient of a whole network by hand is possible for tiny cases and hopeless for real ones; and every time someone changes the network, the derivation would have to be redone. We need derivatives computed automatically, for any formula.</p>,
      },
      {
        heading: 'The idea: a graph of tiny steps',
        body: <>
          <p>Any formula is built from a few simple operations. <code>d = a·b + a</code> is really two steps:</p>
          <p className="formula">c = a · b        d = c + a</p>
          <p>Drawn as boxes and arrows, this is a <b>computation graph</b>. Every box is an operation whose own derivative is simple and known in advance: for a multiply, <code>∂(a·b)/∂a = b</code>; for an add, 1; for tanh, <code>1 − out²</code>. These are the <b>local derivatives</b>.</p>
        </>,
      },
      {
        heading: 'The chain rule',
        body: <>
          <p>Leibniz’s chain rule (1676) says how local slopes combine: if <code>d</code> depends on <code>c</code>, and <code>c</code> on <code>a</code>, then</p>
          <p className="formula">∂d/∂a (through c) = ∂d/∂c × ∂c/∂a</p>
          <p>Multiply the local slopes along a path. Intuitively: if <code>d</code> moves 1 unit per unit of <code>c</code>, and <code>c</code> moves 3 units per unit of <code>a</code>, then <code>d</code> moves 3 units per unit of <code>a</code>.</p>
          <p>And if a value is used in several places, as <code>a</code> is here, its effects <b>add up</b> over all paths: 1 directly through the add, plus 1 × 3 through the multiply, gives 4.</p>
        </>,
      },
      {
        heading: 'Why this matters',
        body: <p>With these two rules (multiply along a path, add over paths) a program can differentiate any formula made of known operations, of any size, without a person deriving anything. This is <b>automatic differentiation</b>. Adding a new operation only requires writing its one local derivative.</p>,
      },
    ],
    faq: [
      { q: 'Is this the same as symbolic differentiation (like a computer algebra system)?', a: <>No. Symbolic differentiation produces a formula for the derivative, which can grow enormous. Automatic differentiation computes the derivative’s <i>value</i> at the current numbers, step by step, and stays as cheap as the original formula.</> },
      { q: 'Who builds the graph in PyTorch?', a: <>PyTorch records each operation as your Python code runs it (“define by run”), building the graph on the fly. This site builds the graph once from a description; the arithmetic is the same.</> },
      { q: 'Why add over paths and not multiply?', a: <>Each path is a separate route by which a nudge reaches the output; their small effects stack up, just as two pushes on a door add.</> },
    ],
    labs: [{
      preset: 'expression', hash: 'pass=reverse', label: 'Multiply along paths, add over paths',
      steps: [<>Press ▶ (one step) slowly. Each backward step shows the multiplication, e.g. <code>1.00 × 3.00 = 3.00</code>.</>, <>Watch <code>a</code>: it receives 1, then 3 more: “sum of 2 pieces”.</>],
    }],
  },

  {
    id: 'backprop',
    level: 'beginner',
    title: 'Backpropagation',
    summary: 'Every gradient in one backward sweep: the algorithm that made deep learning practical.',
    question: <>Nudging each knob costs one run per knob. How do networks with billions of knobs get all their gradients at once?</>,
    sections: [
      {
        heading: 'Before this',
        body: <>
          <p>The nudge-and-measure method needs one run of the network per parameter. For the roughly 1,200 parameters of this site’s largest example, that is 1,200 runs per training step; for a modern model, billions. Through the 1960s and 70s nobody had an efficient, general way to train networks with hidden layers, and research on them largely stalled.</p>
          <p>The fix was found several times: Seppo Linnainmaa described reverse-mode automatic differentiation in 1970; Paul Werbos proposed using it for neural networks in 1974; and David Rumelhart, Geoffrey Hinton and Ronald Williams made it famous in 1986 by showing it let multi-layer networks learn useful internal features.</p>
        </>,
      },
      {
        heading: 'The idea',
        body: <>
          <p>Apply the chain rule <b>backwards</b>, starting from the loss. The loss’s gradient with respect to itself is 1. Then visit the boxes from last to first; each box takes the gradient it has received and passes back, to each of its inputs, that gradient times its local derivative:</p>
          <p className="formula">∂L/∂input += ∂L/∂box × ∂box/∂input</p>
          <p>When the sweep reaches the start, every box, and therefore every parameter, has its gradient.</p>
        </>,
      },
      {
        heading: 'Why it is so cheap',
        body: <p>Every box is visited once going forward and once going back. So one backward pass costs about the same as a couple of forward passes, <b>no matter how many parameters there are</b>. All the gradients for the price of roughly two or three evaluations, instead of one evaluation per parameter. This one fact is why training large networks is possible at all.</p>,
      },
      {
        heading: 'The order matters',
        body: <p>A box may only pass its gradient on once it has received <i>all</i> of it, from every place its output is used; otherwise it would pass on an incomplete number. The stepper’s <i>Backward order</i> panel shows this: a box is “waiting” until all its pieces have arrived, then “ready”. PyTorch’s engine keeps exactly this kind of ready queue.</p>,
      },
      {
        heading: 'The price: memory',
        body: <p>The backward pass needs values computed during the forward pass (the multiply needs <code>a</code> and <code>b</code> to know its slopes). So they are kept in memory until the backward pass is done. For big models, this stored memory, not the arithmetic, is often the limit; tricks like activation checkpointing trade recomputation for memory.</p>,
      },
      {
        heading: 'Where it is used',
        body: <p><code>loss.backward()</code> in PyTorch, <code>jax.grad</code>, TensorFlow’s <code>GradientTape</code>: every deep-learning framework is, at its core, an implementation of this sweep, plus fast versions of the boxes.</p>,
      },
    ],
    faq: [
      { q: 'Is backpropagation the same thing as gradient descent?', a: <>No. Backpropagation <i>computes</i> the gradient. Gradient descent (or Adam, etc.) <i>uses</i> it to update the knobs. A train step is: forward, backward (backprop), update.</> },
      { q: 'Why is it called “reverse mode”?', a: <>Because derivatives flow from the output back toward the inputs. There is also a forward mode, which carries derivatives the other way; it has its own lesson, with the reason training uses reverse.</> },
      { q: 'Do the gradients from backprop match the nudge method?', a: <>Yes, up to rounding. This site’s engine checks every operation against nudge-and-measure, and against PyTorch.</> },
      { q: 'Does the brain do backpropagation?', a: <>Probably not in this form; whether brains do something equivalent is an open research question.</> },
    ],
    labs: [
      {
        preset: 'expression', hash: 'pass=reverse', label: 'Watch the sweep and the ready queue',
        steps: [<>Press ▶ (one step) until the backward steps start, and follow the <i>Backward order</i> panel: <code>c</code> becomes ready only after <code>d</code> has sent; <code>a</code> waits for 2 pieces.</>],
      },
      {
        preset: 'mlp', hash: 'pass=reverse&zoom=scalar', label: 'A real network, one number at a time',
        steps: [<>Press Play and watch gradients flow back through about 240 single-number boxes.</>, <>Open <i>Reverse vs forward mode</i>: one backward pass gave all 22 derivatives that forward mode needed 22 passes for.</>],
      },
    ],
  },
];
