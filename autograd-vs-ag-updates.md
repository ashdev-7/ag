# autograd-engine vs ag: what changed

Two repositories:

| | **autograd-engine** (the original) | **ag** (this repository) |
|---|---|---|
| GitHub | github.com/ashdev-7/autograd-engine | github.com/ashdev-7/ag |
| Local | `New folder/autograd-engine` | `New folder/ag` |
| State | One commit, unchanged since ag began | 57 commits; engine, website, tests, deployment |
| Live | — | https://ag-glass-box.vercel.app |

ag's engine is **a rewrite**, not an edit of the original. The original was kept
exactly as it was (entry 5 in `product-dev/progress.txt`): ag's files were
moved out of it, its one `CMakeLists.txt` change was reverted, and it is still at
its single commit. The only file there that is not committed is your own
`summary.txt`, left alone.

All numbers below were measured on 2026-10-16 on this machine (g++ 16.1, `-O2`).

---

## 1. The short answer

| | autograd-engine | ag |
|---|---|---|
| What a value is | one number (`double`); "tensors" store a vector but every op reads element `[0]` | real tensors: up to 4 dimensions, `float32`, numpy broadcasting |
| Operations | 8 | 24 |
| Gradients | reverse mode | reverse mode **and** forward mode, for every op |
| Training | none: no optimizer, no data, no loss functions | 6 optimizers, 4 learning-rate schedules, gradient clipping, 8 initialisers, 6 datasets, batching |
| Same small network, one training step | 17,500 µs | **57 µs** (about **300×** faster) |
| Memory over a training run | grows **9 MB per step** (1.5 GB after 3 s) and is never freed | **flat** (5 MB) |
| A chain of 100,000 operations | **crashes** (stack overflow) | no recursion anywhere; the graph is a flat list |
| `.backward()` called twice | wrong gradients (7 and 4 instead of 4 and 2) | every pass starts from zero; seeking back and forth is exact |
| Errors | throws C++ exceptions (division by zero, log of ≤ 0) | never throws: every call returns a status naming the node and input where it failed |
| NaN / infinity | spreads silently | detected at the node where it first appears; the training step is rolled back |
| Checked against | 44 hand-written expected values | finite differences, its own forward mode, and **PyTorch 2.9** (39 comparisons, worst relative error 2e-5) |
| Runs in | Python (pybind11) and C++ | C++, WebAssembly, and plain JavaScript (for browsers without WebAssembly) |
| Lines of code | 344 engine + 276 tests | 3,871 engine + 1,512 tests and benchmark |
| Third-party code | pybind11 | none |

---

## 2. Are the PyTorch-like functions still there?

**The Python interface is gone; PyTorch's behaviour is now checked, not imitated.**

The original's whole user interface was a pybind11 module, `import autograd`,
with a PyTorch-flavoured `Tensor`:

```python
a = autograd.Tensor(2.0)
b = autograd.Tensor(3.0)
d = a * b + a
d.backward()
a.grad, b.grad          # [4.0], [2.0]
# also: - / .pow(n) .exp() .log() .relu() .zero_grad() .value .data .op .node_id
```

ag has **no Python module**. Its users are web pages, so its interface is a
C API of 41 functions (`engine/include/ag/capi.h`) that the browser calls through
WebAssembly. Instead of looking like PyTorch, ag is **tested against PyTorch**:

- `tests/torch_check.py` rebuilds every op case in real PyTorch (float32) from a
  JSON dump of the engine's values and gradients, and every optimizer for 10
  steps against `torch.optim`. Today: 39 checks, all pass.
- Its semantics follow PyTorch on purpose: numpy broadcasting, `x @ W` for
  matrices stored `[in, out]`, `detach`, a `noGrad` flag that behaves like
  `torch.no_grad()`, `softmax` with an axis, `mse_loss` and
  `softmax_cross_entropy` matching `F.mse_loss` and `F.cross_entropy`, and
  SGD, momentum, Nesterov, RMSprop, Adam and AdamW matching `torch.optim`.

What the original's Python functions became in ag:

| Original (Python) | ag |
|---|---|
| `Tensor(2.0)`, `Tensor([...])` | a graph node: `Input`, `Param` or `Const`, with a shape |
| `a + b`, `a - b`, `a * b`, `a / b` | `Add`, `Sub`, `Mul`, `Div`, now on whole tensors with broadcasting |
| `.pow(n)`, `.exp()`, `.log()`, `.relu()` | `Pow`, `Exp`, `Log`, `Relu` |
| `.backward()` | a Reverse pass (or a Train pass, which also updates the weights) |
| `.grad` | `ag_session_read(..., grad)`, or summary statistics without copying |
| `.zero_grad()` | automatic: every pass starts from zero |
| `.value`, `.data` | `ag_session_read(..., value)` |
| `.op`, `.node_id`, `.node_type`, `set_node_type` | the op table (name, symbol, category) and each node's `name` and `module` |
| `__repr__` | `dump(graph)`, a stable text form |

To use ag from Python again, the C API is plain C and could be wrapped with
`ctypes` or pybind11 in about a day. Nothing in the product needs it, so it was not built.

---

## 3. How the engine works now: the design change

The original is **define-by-run**, like PyTorch's eager mode. Every operation
immediately allocates a node with `make_shared`, stores a `std::function`
closure for its derivative, and links to its inputs; `backward()` sorts the
nodes recursively and calls each closure.

ag is **define-then-run**, like `torch.compile`, JAX's `jit` or TensorFlow 1:

1. **Graph** (`graph.hpp`): plain data. A node may only use inputs with a
   smaller id, so the list is already in topological order and a cycle
   cannot even be written down. No recursive sort is needed.
2. **Compile** (`program.hpp`): for one batch size, checks every shape, works
   out which nodes need gradients, lays out every value, gradient and tangent
   buffer in one block, and writes four **schedules**: Forward, Reverse,
   Train and ForwardMode. A schedule is a flat list of micro-steps (compute
   this value; seed the loss gradient; push gradient from this node into
   input k; update this parameter). The website's step-through view walks
   exactly these lists.
3. **Execute** (`exec.hpp`): one arena allocated once; after that a training
   step allocates nothing. The cursor can advance one step, run to the end, or
   seek to any step by replaying from the start, which is why going backwards
   in the site is exact.
4. **Session** (`session.hpp`): a training run. Two compiled programs (a
   training batch and an evaluation batch) share one set of parameters, plus
   the optimizer, the dataset and the batcher. Every training step is a
   transaction: state is snapshotted and restored if a NaN appears.

Every op is one row in a table (`ops.hpp`) with five functions: shape
inference, forward, VJP (reverse mode), JVP (forward mode), and **scalarize**,
which expands the op into single-number nodes for the site's "single numbers"
zoom. The scalar view is run by the same engine, so it cannot disagree with the
tensor view about a derivative.

---

## 4. Every change, by area

### Values and shapes
- **Original:** `vector<double>`, but every op reads and writes only `[0]`.
  Its own notes say: "It is a scalar engine with vector storage."
- **ag:** tensors of rank 0 to 4, row-major, `float32`. Numpy broadcasting with
  the gradient summed back over broadcast axes. A batch dimension is resolved
  at compile time. Sums are taken in `double` and converted with saturation,
  so an overflowing sum becomes infinity, never undefined behaviour.

### Operations (8 → 24)
| Group | Original | ag |
|---|---|---|
| Sources | leaf | `Input`, `Param`, `Const` |
| Arithmetic | add, sub, mul, div, pow, exp, log | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Pow`, `Exp`, `Log` |
| Linear algebra | — | `MatMul` |
| Reductions | — | `Sum`, `Mean` (any axis, keepdim) |
| Shape | — | `Reshape`, `Transpose` |
| Activations | relu | `Relu`, `LeakyRelu`, `Tanh`, `Sigmoid`, `Softmax` |
| Losses | — | `MseLoss`, `SoftmaxCrossEntropy` |
| Gradient control | — | `Detach`, plus a `noGrad` flag on any node |

### Differentiation
- **Original:** reverse mode only, through one closure per node.
- **ag:** reverse mode (VJP) and forward mode (JVP) for every op, checked
  against each other. Gradients stop at `Detach` and `noGrad` in both modes and
  in the single-number view.

### Training (did not exist)
- **Optimizers:** SGD, momentum, Nesterov, RMSprop, Adam, AdamW. Their state
  (momentum, second moment) is readable per weight.
- **Learning-rate schedules:** constant, step decay, cosine, warmup + cosine.
- **Global-norm gradient clipping**, and weight decay (L2, or decoupled for AdamW).
- **Initialisers:** zeros, constant, uniform, normal, Xavier, He (uniform and normal).
- **Datasets:** moons, circles, spirals, XOR, Gaussians, 1-D sine, generated
  from a seed, with a train/validation split and shuffled batches.
- **Determinism:** an own PCG32 random generator, so a seed gives the same
  numbers natively, in WebAssembly and in JavaScript. The standard library's
  distributions differ between compilers and are not used.

### Reading what happens (for the site)
- Per-node statistics without copying tensors: min, max, mean, std, L2 norm,
  share of zeros (dead units), NaN count, a 32-bin histogram.
- Decision-boundary grids, loss surfaces over two weights (the optimizer race),
  train and validation loss, and export/import of all parameters so two
  optimizers can start from identical weights.

### Memory and lifetime
- **Original:** every op node is a `shared_ptr` whose closure holds a
  `shared_ptr` to itself, so **no op node is ever freed**. The original's notes
  call this out. Measured: +9 MB per training step.
- **ag:** no per-node allocation. One arena per compiled program, allocated
  with `calloc` so running out of memory returns `OutOfMemory` instead of
  aborting the WebAssembly module. Measured: memory flat over 50,000 steps.

### Errors and robustness
| Situation | Original | ag |
|---|---|---|
| Division by zero, log of ≤ 0 | throws an exception | the value becomes inf/NaN and is reported at that node; the step is rolled back |
| A NaN appears mid-training | spreads into every weight | stops at the first node where it appears, names it, restores the weights |
| Very deep graph | recursive sort: stack overflow at 100,000 ops (measured) | no recursion |
| `backward()` twice | intermediate gradients add up: 7 and 4 instead of 4 and 2 (measured) | every pass zeroes all gradients and values first |
| `backward(nullptr)` | dereferences null | stale or freed handles return `NotFound`; handles carry a generation number |
| Shapes too large | — | hard limits (nodes, elements, parameters, memory) checked at compile time; a shape whose size overflows is refused |
| Bad attributes (NaN exponent etc.) | — | refused at the node |
| Threads | not thread-safe | not thread-safe either (the C API keeps one handle table); the site gives each worker its own engine instance |

### Testing
| | Original | ag |
|---|---|---|
| Checks | 44 hand-computed values | 175 engine checks + 33 skeleton checks |
| Exit code on failure | always 0 (a failure only prints FAIL) | non-zero |
| Finite differences | not implemented | every op, both modes |
| Forward vs reverse mode | — | every op |
| Against PyTorch | — | 39 cases: every op and 6 optimizers over 10 steps |
| Undefined-behaviour checks | — | the whole suite also runs with UB traps |
| Builds compared | — | native vs WebAssembly vs JavaScript: batch losses over 1,500 steps agree to 1.4e-7 |
| Scalar view | — | every op case is also expanded to single numbers and must give the same loss and gradients |

### Builds and platforms
- **Original:** CMake, fetches pybind11, builds a Python extension and a test binary.
- **ag:** a Makefile (CMake kept for IDEs). Native tests, a benchmark,
  **WebAssembly** (`ag.wasm`, 176 KB) and a **plain-JavaScript** build
  (`ag-js.mjs`) for browsers without WebAssembly. Every function in `capi.h`
  is exported and nothing else. No third-party code.

### Speed (ag's own benchmark, Adam, tanh MLP on moons)
| Network | Batch | Weights | µs per training step |
|---|---|---|---|
| 2 layers × 16 | 32 | 352 | 83 |
| 2 × 64 | 32 | 4,480 | 649 |
| 2 × 128 | 32 | 17,152 | 2,090 |
| 4 × 128 | 128 | 50,176 | 25,747 |

In the browser, WebAssembly runs 1.4 to 2.3 times slower than native, and every
lab on the site trains in 0.07 to 0.4 ms per step.

**Head-to-head** (both at `-O2`): MLP 2 → 16 → 16 → output, ReLU, mean squared
error, batch of 32, SGD at 0.05, timed over 3 seconds.

| | Original | ag |
|---|---|---|
| Time per step | 17,502 µs | 57 µs |
| Graph per step | 20,609 scalar nodes, rebuilt every step | 17 tensor nodes, compiled once |
| Memory | 52 MB → 1,580 MB in 172 steps | 5 MB → 5 MB over 52,000 steps |

The original builds a fresh node, `shared_ptr` and `std::function` for every
multiply and add of every sample on every step. ag runs fixed loops over
arrays it allocated once.

---

## 5. What the original had that ag does not

- **The Python interface** (`import autograd`), described in section 2.
- **`double` precision.** ag computes in `float32`, like PyTorch's default and
  like GPUs, and accumulates sums in `double`.
- **Define-by-run.** Python `if`s and loops could shape the graph as it ran. In
  ag a graph is fixed when it is compiled, which is what makes the schedules,
  exact seeking and zero allocation possible.
- **`parents` back-pointers** (`weak_ptr`). ag counts each node's consumers at
  compile time instead.

Everything else the original did, ag does.

---

## 6. Changes to the autograd-engine repository

None. It is at its single commit (`569033d initial commit`). Its tests still
build and pass, 44 of 44, re-run today. ag was developed in this separate
repository so the original stays as it was.

---

## 7. What else ag added: the website

The engine exists to power https://ag-glass-box.vercel.app:

- **Lessons:** 17 short lessons, each with live figures and labs.
- **Step-through labs:** five graphs (an expression, the same with `detach`, one
  neuron, broadcasting, a two-layer MLP). Every forward value, gradient
  contribution and update is one step; scrub back and forth; switch between
  tensors and single numbers; compare reverse with forward mode; see the
  backward order the way PyTorch's ready queue works.
- **Training labs:** moons, circles, spirals, XOR, vanishing gradients, dead
  ReLUs, overfitting and its cure. Each shows the decision boundary, the loss
  (train and test), how hard each layer is corrected, each layer's outputs,
  the neuron diagram, and any weight's optimizer state. Two optimizers can
  train side by side from identical weights.
- **Optimizer race:** SGD, momentum, RMSprop and Adam on three loss surfaces.
- **Works everywhere:** phones and tablets, and browsers without workers or
  without WebAssembly (falls back to running on the page or to the JavaScript
  build). A crashed engine restarts silently; any page that fails to draw
  offers "Start it again".
- **Tests:** unit tests; a browser test in Chromium, Firefox and WebKit; a
  164-page click-through, also under six simulated crippled browsers; a
  checker that follows every lesson instruction; a test that uses every
  control on every page on desktop and on phones; and a phone responsiveness
  check.
- **Deployment:** Vercel, with a link preview, a security policy and visitor
  analytics.

The full history, request by request, is in `product-dev/progress.txt`.
