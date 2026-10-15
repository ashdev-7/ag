import { ActivationWidget } from '../../components/widgets.tsx';
import type { Lesson } from './types.ts';

export const INTERMEDIATE: Lesson[] = [
  {
    id: 'tensors',
    level: 'intermediate',
    title: 'Tensors and broadcasting',
    summary: 'Why networks work on whole arrays at once, and how gradients flow through a shape mismatch.',
    question: <>Real networks don’t compute one number at a time. What are “tensors”, and what happens to gradients when arrays of different shapes are combined?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>A layer with 64 neurons and 64 inputs does 4,096 multiplications per example, and training uses many examples at once. Written as loops over single numbers this is slow and unreadable. Hardware (vector units, GPUs) is built to do the same operation on many numbers at the same time, if the numbers are laid out together.</p>,
      },
      {
        heading: 'The idea: arrays as values',
        body: <>
          <p>A <b>tensor</b> is an array of numbers with a <b>shape</b>: a single number has shape <code>[]</code>, a list <code>[3]</code>, a table <code>[3, 2]</code> (3 rows, 2 columns). A whole layer then becomes one operation on tables:</p>
          <p className="formula">outputs[batch, 64] = inputs[batch, 64] @ W[64, 64] + b[64]</p>
          <p>where <code>@</code> is matrix multiplication and each row is one example. The first dimension, the <b>batch</b>, lets one operation process many examples at once.</p>
        </>,
      },
      {
        heading: 'Broadcasting',
        body: <>
          <p>In the line above, <code>b</code> has 64 numbers but the product has <code>batch × 64</code>. The rule NumPy made standard: a dimension of size 1 (or a missing one) is <b>stretched</b> to match, so the same bias row is added to every example.</p>
          <p>Backwards, the stretch must be undone. Each bias number was used once per row, and a value used several times receives the <b>sum</b> of its gradients (the chain-rule lesson). So the bias gradient is the gradient of the output <b>summed over the rows</b>. Get this wrong and the bias gradient comes out the wrong size or too small, a classic bug in hand-written networks.</p>
        </>,
      },
      {
        heading: 'Where it is used',
        body: <p>Everywhere: biases, normalisation layers, attention masks, adding a learning-rate schedule to every weight. Every framework follows the same broadcasting rule, and every framework sums the gradient back over the stretched dimensions.</p>,
      },
    ],
    faq: [
      { q: 'Is a tensor anything more than an array?', a: <>In physics the word means more, but in deep learning it simply means an n-dimensional array of numbers with a shape.</> },
      { q: 'Why does the batch not change the gradient’s shape?', a: <>A parameter has the same shape whatever the batch size; its gradient is summed (or averaged, through a mean in the loss) over the examples in the batch.</> },
      { q: 'What does “0.12 ± 0.40” on a box mean?', a: <>The average of all the numbers in that tensor, ± their typical spread around it (the standard deviation): a two-number summary of an array too big to print. Click the box to see every number.</> },
    ],
    labs: [{
      preset: 'broadcast', hash: 'pass=reverse', label: 'A bias added to every row',
      steps: [<>Press End, then click <code>x + b</code>: its gradient is a 3 × 2 grid.</>, <>Click <code>b</code>: its 2 numbers are the column sums of that grid.</>, <>Switch to <i>Single numbers</i> to see each <code>b</code> element feeding three adds.</>],
    }],
  },

  {
    id: 'nonlinearity',
    level: 'intermediate',
    title: 'Why networks need non-linearity',
    summary: 'The perceptron, the XOR problem, and why every layer ends in a bend.',
    question: <>Why does every neuron pass its sum through tanh, sigmoid or ReLU? Wouldn’t a stack of weighted sums be simpler?</>,
    sections: [
      {
        heading: 'Before this: the perceptron and its limit',
        body: <>
          <p>Rosenblatt’s perceptron (1958) was a single layer: weighted sum, then a threshold. It learned to separate classes whenever <b>one straight line</b> could separate them. In 1969 Marvin Minsky and Seymour Papert published a careful analysis of what single-layer perceptrons cannot do; the most famous example is <b>XOR</b>: points in opposite corners belong to the same class. No single straight line separates them. The book contributed to a long decline in neural-network research.</p>
        </>,
      },
      {
        heading: 'Why stacking plain layers does not help',
        body: <>
          <p>A layer without a bend computes <code>x @ W + b</code>, a linear function. Two such layers compute <code>(x @ W₁ + b₁) @ W₂ + b₂</code>, which simplifies to <code>x @ (W₁W₂) + (b₁W₂ + b₂)</code>: again a single linear layer. Any number of stacked linear layers collapses into one, and can still only draw straight lines.</p>
        </>,
      },
      {
        heading: 'The idea: bend after every layer',
        body: <>
          <p>Putting a non-linear <b>activation function</b> after each layer stops the collapse. Each hidden neuron now carves the input space with one soft line, and the next layer combines those pieces into curves and closed regions. With one hidden layer, XOR becomes easy. The catch, finding the weights for hidden layers, was solved by backpropagation.</p>
          <ActivationWidget />
        </>,
      },
      {
        heading: 'Which bend?',
        body: <ul>
          <li><b>Sigmoid</b> <code>1/(1 + e^−x)</code>: squashes to (0, 1); the classic choice, and the source of vanishing gradients (advanced lesson).</li>
          <li><b>tanh</b>: squashes to (−1, 1), centred on zero, which usually trains faster than sigmoid.</li>
          <li><b>ReLU</b> <code>max(0, x)</code>: zero for negatives, identity for positives. Cheap, and its slope is exactly 1 where active, so gradients pass undiminished. The default in most networks since about 2011–2012.</li>
        </ul>,
      },
    ],
    faq: [
      { q: 'Is the activation function learned?', a: <>Usually not; it is a fixed choice. Only the weights and biases around it are learned.</> },
      { q: 'Why not use a very bendy function, like a high power?', a: <>Its slopes would be huge in some places and tiny in others, which makes gradient descent unstable. Good activations are simple, monotone and have well-behaved slopes.</> },
      { q: 'Does the output layer need an activation too?', a: <>Not a bend for its own sake. For classification it ends in softmax (next lesson); for predicting a number it usually ends with no activation at all.</> },
    ],
    labs: [{
      preset: 'xor', label: 'XOR with one hidden layer',
      steps: [<>Press Run. The decision boundary starts as nearly one straight line and bends into a shape that puts opposite corners together, something no single line can do.</>, <>Watch the loss wobble before it settles: this run uses momentum.</>],
    }],
  },

  {
    id: 'classification',
    level: 'intermediate',
    title: 'Choosing between classes: softmax and cross-entropy',
    summary: 'Turning scores into probabilities, and the loss that punishes confident mistakes.',
    question: <>For “which class is this?”, what does the network output, and why don’t we just use squared error?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>A classifier must say how likely each class is. The last layer outputs one raw score per class (a <b>logit</b>); scores can be any number, positive or negative. We need probabilities: positive and adding up to 1.</p>,
      },
      {
        heading: 'Softmax',
        body: <>
          <p className="formula">p_k = e^(z_k) / (e^(z_1) + … + e^(z_K))</p>
          <p>Exponentiating makes every score positive, and larger scores much larger; dividing by the total makes them add up to 1. The biggest score gets the biggest probability, but every class keeps some. It is a smooth (“soft”) version of picking the maximum, hence the name. The same formula appears in statistics (multinomial logistic regression) and physics (the Boltzmann distribution).</p>
        </>,
      },
      {
        heading: 'Cross-entropy',
        body: <>
          <p>For one example whose true class is <code>c</code>:</p>
          <p className="formula">loss = −log( p_c )</p>
          <p>If the network gives the right class probability 1, the loss is 0. Probability 0.5 costs 0.69 (that is <code>ln 2</code>: a coin flip between two classes). Probability 0.01 costs 4.6. The loss is the average over the batch. The name comes from information theory (Claude Shannon, 1948): it measures how surprised the model is by the true answer.</p>
        </>,
      },
      {
        heading: 'Why not squared error?',
        body: <>
          <p>Through a softmax, squared error has a flaw: when the network is <b>confidently wrong</b> (right class near probability 0), the softmax is flat there and squared error’s gradient becomes tiny. The model learns slowest exactly when it is most wrong.</p>
          <p>Cross-entropy cancels that flatness. Its gradient with respect to the scores is simply</p>
          <p className="formula">∂loss/∂z_k = p_k − t_k</p>
          <p>predicted probability minus target: large when badly wrong, zero when right. Simple, and strong where it is needed.</p>
        </>,
      },
      {
        heading: 'Where it is used',
        body: <p>Image classifiers, spam filters, and language models, which are classifiers over the vocabulary: “which word comes next?” is a softmax over tens of thousands of classes, trained with exactly this loss.</p>,
      },
    ],
    faq: [
      { q: 'Why are the probabilities computed with “log-sum-exp”?', a: <>Exponentials overflow quickly (<code>e^1000</code> is infinite in float32). Engines subtract the largest score first, which does not change the result but keeps the numbers small. This site’s engine does that.</> },
      { q: 'If p is not used by the loss, why is it in the graph?', a: <>The loss takes the logits directly and does the softmax inside, which is more accurate. The separate <code>p</code> box is only there to read the probabilities and draw the decision boundary.</> },
      { q: 'What is a good loss value?', a: <>Compare with 0.69, the coin flip for two classes. Below it, the network knows something; near 0, it is confident and right on those examples.</> },
    ],
    labs: [
      {
        preset: 'mlp', hash: 'pass=reverse', label: 'Softmax and cross-entropy, step by step',
        steps: [<>Press End. Click <code>logits</code> and <code>p</code> to compare raw scores with probabilities.</>, <>Click <code>logits</code> again after the backward pass: its gradient is <code>(p − y) / 4</code>, divided by 4 because the loss averages over 4 points.</>],
      },
      {
        preset: 'moons', label: 'Watch the loss leave 0.69',
        steps: [<>Press Run: the loss starts near 0.69 (a guess) and falls as the boundary bends around the moons.</>],
      },
    ],
  },

  {
    id: 'training-loop',
    level: 'intermediate',
    title: 'The training loop: batches, epochs and overfitting',
    summary: 'Why we train on small random batches, and how to tell learning from memorising.',
    question: <>We have a loss and a way to step downhill. What does a full training run actually look like, and how do we know the network learned something real?</>,
    sections: [
      {
        heading: 'The problem with the whole dataset',
        body: <p>The true loss is the average over all training examples. Computing its exact gradient means a forward and backward pass over the entire dataset for every single step: millions of examples per step for real datasets. Far too slow.</p>,
      },
      {
        heading: 'The idea: estimate the gradient from a small batch',
        body: <>
          <p>Herbert Robbins and Sutton Monro showed in 1951 that stepping along a <b>noisy estimate</b> of the gradient still converges, as long as the steps shrink appropriately. So each step uses a small random <b>batch</b> (here, 16 points): cheap, slightly noisy, and on average pointing the right way. This is <b>stochastic gradient descent</b>.</p>
          <ul>
            <li>One <b>step</b>: take the next batch, forward, backward, update.</li>
            <li>One <b>epoch</b>: enough steps to see every training example once. The data are shuffled between epochs.</li>
          </ul>
          <p>The noise is not only a cost: it shakes the model out of shallow dips and flat regions.</p>
        </>,
      },
      {
        heading: 'Learning versus memorising',
        body: <>
          <p>A network with enough knobs can drive the loss on its training examples to zero by memorising them, and still be useless on new data. This is <b>overfitting</b>. The only way to tell is to keep some examples <b>out</b> of training and measure the loss on them: the <b>validation set</b> (here 20% of the points).</p>
          <ul>
            <li>Both losses fall: it is learning the pattern.</li>
            <li>Training loss falls but validation loss rises: it has started memorising. Stopping there is called <i>early stopping</i>.</li>
          </ul>
        </>,
      },
      {
        heading: 'Where it is used',
        body: <p>Every training run is this loop. Large models train for many epochs over billions of examples with batches of thousands; the structure is identical. Reported results always come from data the model did not train on.</p>,
      },
    ],
    faq: [
      { q: 'Why is the solid loss line so jumpy?', a: <>Each point is the loss of one batch of 16, and batches differ: some are easy, some hard. The dashed validation line averages over all held-out points, so it is smooth.</> },
      { q: 'How big should a batch be?', a: <>Bigger batches give smoother gradients and use hardware better, but each example then contributes less learning per unit of compute. Common sizes range from 16 to several thousand.</> },
      { q: 'Why don’t these examples overfit much?', a: <>The networks are small for the amount of data and the data have a simple shape. Overfitting shows most with many parameters and few examples.</> },
    ],
    labs: [{
      preset: 'moons', label: 'Batches and validation',
      steps: [<>Press Run and compare the solid (batch) and dashed (held-out) loss lines.</>, <>Press Reset, then Step a few times: each Step is exactly one batch, one forward, one backward, one update.</>],
    }],
  },

  {
    id: 'modes',
    level: 'intermediate',
    title: 'Forward mode versus reverse mode',
    summary: 'Two ways to compute derivatives, and why training always uses the backward one.',
    question: <>Backpropagation goes backwards. Is there a forward way to get derivatives, and if so, why isn’t it used for training?</>,
    sections: [
      {
        heading: 'Forward mode',
        body: <>
          <p>Forward-mode automatic differentiation (described by Robert Wengert in 1964) carries a derivative <b>along with</b> each value as the forward pass runs. Choose one input and give it a “nudge” <code>ṫ = 1</code>, every other input 0. Each box computes its value and also how much its value moves:</p>
          <p className="formula">ṫ_box = Σ (∂box/∂input) × ṫ_input</p>
          <p>When the pass reaches the loss, its <code>ṫ</code> is <code>∂L/∂(chosen input)</code>. No stored values, no backward sweep.</p>
        </>,
      },
      {
        heading: 'The cost difference',
        body: <>
          <p>One forward-mode pass gives the derivative of <b>every output</b> with respect to <b>one input</b>. One reverse-mode pass gives the derivative of <b>one output</b> with respect to <b>every input</b>.</p>
          <ul>
            <li>Training: one output (the loss), millions of inputs (the parameters). Reverse mode: 1 pass. Forward mode: millions.</li>
            <li>A simulation with 2 adjustable settings and 10,000 outputs: forward mode needs 2 passes; reverse mode 10,000.</li>
          </ul>
          <p>Same derivatives either way; only the cost differs, and it depends on the shape of the problem.</p>
        </>,
      },
      {
        heading: 'Where each is used',
        body: <ul>
          <li><b>Reverse mode:</b> all neural-network training (<code>loss.backward()</code>).</li>
          <li><b>Forward mode:</b> sensitivity analysis in science and engineering, few-input problems, and combined with reverse mode to get second derivatives cheaply. JAX exposes both as <code>jvp</code> (forward) and <code>vjp</code> (reverse); PyTorch has <code>torch.func.jvp</code>.</li>
        </ul>,
      },
    ],
    faq: [
      { q: 'Why is forward mode not just the nudge method?', a: <>The nudge method measures the change numerically and is approximate. Forward mode carries the exact derivative alongside the value using each box’s known local derivative; it is exact up to rounding.</> },
      { q: 'Does forward mode need less memory?', a: <>Yes. It never stores values for later; each box’s value and derivative are used once and can be forgotten. Reverse mode must keep the forward values for the backward sweep.</> },
      { q: 'What are “dual numbers”?', a: <>A neat way to implement forward mode: numbers of the form <code>a + b·ε</code> with <code>ε² = 0</code>. Doing ordinary arithmetic on them carries the derivative in <code>b</code> automatically.</> },
    ],
    labs: [{
      preset: 'expression', hash: 'pass=forwardMode', label: 'The same derivatives, two ways',
      steps: [<>Step through forward mode with direction <code>a</code>: the loss’s tangent ends at 4. Switch direction to <code>b</code>: 2.</>, <>Press <i>Reverse vs forward mode</i>: reverse needed 1 pass for both, forward mode 2. On the two-layer MLP the gap is 1 against 22.</>],
    }],
  },

  {
    id: 'stop-gradient',
    level: 'intermediate',
    title: 'Stopping gradients: detach and no_grad',
    summary: 'Using a value without learning through it, and why that is often exactly what you want.',
    question: <>Sometimes a value should be used but not learned through. When, and how do you tell the engine?</>,
    sections: [
      {
        heading: 'The problem',
        body: <p>By default, a gradient flows back through every operation to every parameter that influenced the loss. Often that is wrong: part of the model is already trained and must not change; or a value is a <i>target</i> to move toward, and moving the target toward the prediction would be cheating; or you are only making predictions and do not need gradients at all.</p>,
      },
      {
        heading: 'The tools',
        body: <ul>
          <li><b><code>detach(x)</code></b>: the same value as <code>x</code>, but treated as a constant by the backward pass. The gradient stops there.</li>
          <li><b><code>no_grad</code></b>: a region where nothing records gradients at all; outputs computed inside are constants for the backward pass. Used for inference, where it also saves the memory that backpropagation would need.</li>
        </ul>,
      },
      {
        heading: 'Where it is used',
        body: <ul>
          <li><b>Fine-tuning:</b> keep a pretrained network fixed and train only a small new part on top.</li>
          <li><b>Reinforcement learning:</b> DQN (2015) computes its learning targets from a separate, frozen copy of the network; the targets are not differentiated.</li>
          <li><b>GANs:</b> when training the discriminator, the generator’s output is detached so the generator is not updated in that step.</li>
          <li><b>Self-supervised learning:</b> methods such as BYOL and SimSiam (2020–21) rely on a stop-gradient on one branch to avoid collapsing to a trivial solution.</li>
          <li><b>Evaluation:</b> every validation loss and prediction runs under <code>no_grad</code>.</li>
        </ul>,
      },
    ],
    faq: [
      { q: 'Does detach change the forward value?', a: <>No. The value is identical; only the backward pass treats it as a constant.</> },
      { q: 'Is a parameter behind a detach still trained?', a: <>Only if it reaches the loss some other way. If every path goes through a detach, it receives no gradient and the optimizer leaves it alone; this site’s engine does not even allocate a gradient for it.</> },
      { q: 'Is stopping gradients the same as a learning rate of 0?', a: <>For that parameter, similar in effect, but detach also stops the gradient flowing further back, and saves the work of computing it.</> },
    ],
    labs: [{
      preset: 'detach', hash: 'pass=reverse', label: 'Cut one path',
      steps: [<>Press End: <code>∂d/∂a = 1</code>, not 4. The path through <code>detach(a)</code> no longer counts.</>, <>Compare with <i>d = a·b + a</i>, where both paths count.</>],
    }],
  },
];
